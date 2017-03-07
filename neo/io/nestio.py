# -*- coding: utf-8 -*-
"""
Classes for reading data files produced by the NEST simulator.

Depends on: numpy, quantities

Supported: Read

Usage:

One can read just a single spiketrain with a certain ID from a file
by using read_spiketrain.
    >>> from neo.io import nestio
    >>> r = nestio.NESTSpikeIO(filename='example_data.gdf')
    >>> st = r.read_spiketrain(gdf_id=8, t_start=0.*pq.ms, t_stop=1000.*pq.ms)
    >>> print(st.magnitude)
    [ 370.6  764.7]
    >>> print(st.annotations)
    {'id': 8}

It is also possible to provide additional annotations for the spiketrain
upon loading the spiketrain
    >>> st = r.read_spiketrain(gdf_id=1, t_start=0.*pq.ms, t_stop=1000.*pq.ms,
    ...                        layer='L6', population='E')
    >>> print(st.annotations)
    {'layer': 'L6', 'id': 1, 'population': 'E'}

One can read multiple spiketrains from a file by passing a list of
IDs to read_segment (or to read_block)
    >>> st = r.read_segment(gdf_id_list=[1,6,8], t_start=0.*pq.ms,
    ...                     t_stop=1000.*pq.ms)
    >>> print(st.spiketrains)
    [<SpikeTrain(array([ 354. ,  603.1]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 274.1]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 370.6, 764.7]) * ms, [0.0 ms, 1000.0 ms])>]


It is also possible to retrieve spiketrains from a file for all neurons
with at least one spike in the defined period
    >>> st = r.read_segment(gdf_id_list=[], t_start=0.*pq.ms,
    ...                     t_stop=1000.*pq.ms)
    [<SpikeTrain(array([ 411.]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 354. ,  603.1]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 691.7]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 274.1]) * ms, [0.0 ms, 1000.0 ms])>,
     <SpikeTrain(array([ 370.6,  764.7]) * ms, [0.0 ms, 1000.0 ms])>]


Authors: Julia Sprenger, Maximilian Schmidt, Johanna Senk, Jakob Jordan, Andrew Davison

"""

# needed for Python 3 compatibility
from __future__ import absolute_import

import os
from datetime import datetime
import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, SpikeTrain, AnalogSignal, ChannelIndex


NEST_UNITS = {
    "V_m": "mV",
    "g_ex": "nS",
    "g_in": "nS"
}


def guess_units(variables):
    return [NEST_UNITS.get(variable, pq.dimensionless)
            for variable in variables]


class NESTSpikeIO(BaseIO):

    """
    Class for reading GDF files, e.g., the spike output of NEST. It handles
    opening the gdf file and reading a block, a segment or single spiketrains.
    """

    # This class can only read data
    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, SpikeTrain]
    readable_objects = [Block, Segment, SpikeTrain]

    has_header = False
    is_streameable = False

    name = 'gdf'
    extensions = ['gdf']
    mode = 'file'

    def __init__(self, filename=None):
        """
        Parameters
        ----------
            filename: string, default=None
                The filename.
        """
        BaseIO.__init__(self, filename=filename)

    def __read_spiketrains(self, gdf_id_list, time_unit,
                           t_start, t_stop, id_column,
                           time_column, **args):
        """
        Internal function called by read_spiketrain() and read_segment().
        """

        # assert that there are spike times in the file
        if time_column is None:
            raise ValueError('Time column is None. No spike times to '
                             'be read in.')

        if gdf_id_list is not None and id_column is None:
            raise ValueError('Specified neuron IDs to '
                             'be ' + str(gdf_id_list) + ','
                             ' but file does not contain neuron IDs.')

        if t_start is None:
            t_start = 0.0 * pq.ms

        if not isinstance(t_start, pq.quantity.Quantity):
            raise TypeError('t_start (%s) is not a quantity.' % (t_start))

        if t_stop is not None and not isinstance(t_stop, pq.quantity.Quantity):
            raise TypeError('t_stop (%s) is not a quantity.' % (t_stop))

        # assert that no single column is assigned twice
        if id_column == time_column:
            raise ValueError('1 or more columns have been specified to '
                             'contain the same data.')

        # load GDF data
        f = open(self.filename)
        # read the first line to check the data type (int or float) of the spike
        # times, assuming that only the column of time stamps may contain
        # floats. then load the whole file accordingly.
        line = f.readline()
        if '.' not in line:
            data = np.loadtxt(self.filename, dtype=np.int32)
        else:
            data = np.loadtxt(self.filename, dtype=np.float)
        f.close()

        # check loaded data and given arguments
        if len(data.shape) < 2 and id_column is not None:
            raise ValueError('File does not contain neuron IDs but '
                             'id_column specified to ' + str(id_column) + '.')

        # get neuron gdf_id_list
        if gdf_id_list is None:
            gdf_id_list = np.unique(data[:, id_column]).astype(int)

        # get consistent dimensions of data
        if len(data.shape) < 2:
            data = data.reshape((-1, 1))

        # use only data from the time interval between t_start and t_stop
        if t_stop is None:
            data = data[np.where(data[:, time_column] >= t_start.rescale(
                            time_unit).magnitude)]
            t_stop = data[:, time_column].max()
        else:
            data = data[np.where(np.logical_and(
                        data[:, time_column] >= t_start.rescale(
                            time_unit).magnitude,
                        data[:, time_column] < t_stop.rescale(time_unit).magnitude))]

        # create an empty list of spike trains and fill in the trains for each
        # GDF ID in gdf_id_list
        spiketrain_list = []
        for i in gdf_id_list:
            # find the spike times for each neuron ID
            if id_column is not None:
                train = data[np.where(data[:, id_column] == i)][:, time_column]
            else:
                train = data[:, time_column]
            # create SpikeTrain objects and annotate them with the neuron ID
            spiketrain_list.append(SpikeTrain(
                train, units=time_unit, t_start=t_start, t_stop=t_stop,
                id=i, **args))
        return spiketrain_list

    def read_block(self, lazy=False, cascade=True,
                   gdf_id_list=None, time_unit=pq.ms, t_start=None,
                   t_stop=None, id_column=0, time_column=1, **args):
        seg = self.read_segment(lazy, cascade, gdf_id_list, time_unit,
                                t_start, t_stop, id_column, time_column, **args)
        blk = Block(file_origin=seg.file_origin, file_datetime=seg.file_datetime)
        blk.segments.append(seg)
        seg.block = blk
        return blk

    def read_segment(self, lazy=False, cascade=True,
                     gdf_id_list=None, time_unit=pq.ms, t_start=None,
                     t_stop=None, id_column=0, time_column=1, **args):
        """
        Read a Segment which contains SpikeTrain(s) with specified neuron IDs
        from the GDF data.

        Parameters
        ----------
        lazy : bool, optional, default: False
        cascade : bool, optional, default: True
        gdf_id_list : list or tuple, default: None
            Can be either list of GDF IDs of which to return SpikeTrain(s) or
            a tuple specifying the range (includes boundaries [start, stop])
            of GDF IDs. Must be specified if the GDF file contains neuron
            IDs, the default None then raises an error. Specify an empty
            list [] to retrieve the spike trains of all neurons with at least
            one spike.
        time_unit : Quantity (time), optional, default: quantities.ms
            The time unit of recorded time stamps.
        t_start : Quantity (time), default: None
            Start time of SpikeTrain. default 0.
        t_stop : Quantity (time), default: None
            Stop time of SpikeTrain. default: time of the last spike
        id_column : int, optional, default: 0
            Column index of neuron IDs.
        time_column : int, optional, default: 1
            Column index of time stamps.

        Returns
        -------
        seg : Segment
            The Segment contains one SpikeTrain for each ID in gdf_id_list.
        """

        if isinstance(gdf_id_list, tuple):
            gdf_id_list = range(gdf_id_list[0], gdf_id_list[1] + 1)

        # create an empty Segment and fill in the spike trains
        seg = Segment(file_origin=self.filename)
        seg.spiketrains = self.__read_spiketrains(gdf_id_list,
                                                  time_unit, t_start,
                                                  t_stop,
                                                  id_column, time_column,
                                                  **args)
        seg.file_datetime = datetime.fromtimestamp(os.stat(self.filename).st_mtime)
        return seg

    def read_spiketrain(
            self, lazy=False, cascade=True, gdf_id=None,
            time_unit=pq.ms, t_start=None, t_stop=None,
            id_column=0, time_column=1, **args):
        """
        Read SpikeTrain with specified neuron ID from the GDF data.

        Parameters
        ----------
        lazy : bool, optional, default: False
        cascade : bool, optional, default: True
        gdf_id : int, default: None
            The GDF ID of the returned SpikeTrain. gdf_id must be specified if
            the GDF file contains neuron IDs.
        time_unit : Quantity (time), optional, default: quantities.ms
            The time unit of recorded time stamps.
        t_start : Quantity (time), default: None
            Start time of SpikeTrain. t_start must be specified.
        t_stop : Quantity (time), default: None
            Stop time of SpikeTrain. t_stop must be specified.
        id_column : int, optional, default: 0
            Column index of neuron IDs.
        time_column : int, optional, default: 1
            Column index of time stamps.

        Returns
        -------
        spiketrain : SpikeTrain
            The requested SpikeTrain object with an annotation 'id'
            corresponding to the gdf_id parameter.
        """

        if gdf_id is None:
            if id_column is not None:
                raise ValueError('No neuron ID specified but file contains '
                                 'neuron IDs in column ' + str(id_column) + '.')
            gdf_ids = None
        elif not isinstance(gdf_id, int):
            raise ValueError('gdf_id has to be of type int or None')
        else:
            gdf_ids = [gdf_id]

        # __read_spiketrains() needs a list of IDs
        return self.__read_spiketrains(gdf_ids, time_unit,
                                       t_start, t_stop,
                                       id_column, time_column,
                                       **args)[0]


class NESTMultimeterIO(BaseIO):
    """
    Class for reading files created by the NEST multimeter.
    """

    # This class can only read data
    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal]
    readable_objects = [Block, Segment, AnalogSignal]

    has_header = False
    is_streameable = False

    name = 'NESTMultimeter'
    extensions = ['dat', 'nmm']
    mode = 'file'

    def read_block(self, lazy=False, cascade=True,
                   variables=None, units=None, **kwargs):
        seg = self.read_segment(lazy=lazy, cascade=cascade,
                                variables=variables,
                                units=units, **kwargs)
        blk = Block(file_origin=seg.file_origin, file_datetime=seg.file_datetime)
        blk.segments.append(seg)
        seg.block = blk
        return blk

    def read_segment(self, lazy=False, cascade=False, variables=None, units=None, **kwargs):
        """


        """
        raw_data = np.loadtxt(self.filename)
        if variables:
            if raw_data.shape[1] != 2 + len(variables):
                raise ValueError("The data file contains {} variables but you have specified {}."
                                 .format(raw_data.shape[1] - 2, len(variables)))
            if units is None:
                units = guess_units(variables)
        else:
            variables = ["variable_{}".format(i) for i in range(raw_data.shape[1]  - 2)]
            if units is None:
                units = [pq.dimensionless for variable in variables]
        sampling_period = (raw_data[1, 1] - raw_data[0, 1]) * pq.ms  # fragile, assumes gid is same for both rows. To fix.
        gids = np.unique(raw_data[:, 0])

        seg = Segment(file_origin=self.filename,
                      file_datetime = datetime.fromtimestamp(os.stat(self.filename).st_mtime))
        n_gids = len(gids)
        for variable, u in zip(variables, units):
            signal = AnalogSignal(np.empty((raw_data.shape[0]//n_gids, n_gids), dtype=float),
                                  units=u,
                                  sampling_period=sampling_period,
                                  name=variable)
            signal.segment = seg
            signal.channel_index = ChannelIndex(np.arange(len(gids)), channel_ids=gids)
            seg.analogsignals.append(signal)
        for i, gid in enumerate(gids):
            for j, signal in enumerate(seg.analogsignals, 2):
                np.ndarray.__setitem__(signal,
                                       (slice(None), i),
                                       raw_data[raw_data[:, 0] == gid, j, np.newaxis])
        return seg
