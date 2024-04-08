"""
Class for reading output files from NEST simulations
( http://www.nest-simulator.org/ ).
Tested with NEST2.10.0

Depends on: numpy, quantities

Supported: Read

Authors: Julia Sprenger, Maximilian Schmidt, Johanna Senk,
Simon Essink, Robin Gutzen, Jasper Albers, Aitor Morales-Gregorio

"""

import os.path
import warnings
from datetime import datetime
import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, SpikeTrain, AnalogSignal

value_type_dict = {"V": pq.mV, "I": pq.pA, 
                   "g": pq.CompoundUnit("10^-9*S"), 
                   "no type": pq.dimensionless}


class NestIO(BaseIO):
    """
    Class for reading NEST output files. GDF files for the spike data and DAT
    files for analog signals are possible.

    Usage:
        >>> from neo.io.nestio import NestIO

        >>> files = ['membrane_voltages-1261-0.dat',
                 'spikes-1258-0.gdf']
        >>> r = NestIO(filenames=files)
        >>> seg = r.read_segment(gid_list=[], t_start=400 * pq.ms,
                             t_stop=600 * pq.ms,
                             id_column_gdf=0, time_column_gdf=1,
                             id_column_dat=0, time_column_dat=1,
                             value_columns_dat=2)
    """

    is_readable = True  # class supports reading, but not writing
    is_writable = False

    supported_objects = [SpikeTrain, AnalogSignal, Segment, Block]
    readable_objects = [SpikeTrain, AnalogSignal, Segment, Block]

    has_header = False
    is_streameable = False

    write_params = None  # writing is not supported

    name = 'nest'
    supported_target_objects = ['SpikeTrain', 'AnalogSignal']
    mode = 'file'

    def __init__(self, filenames=None, target_object='SpikeTrain', **kwargs):
        """
        Parameters
        ----------
            filenames: string or list of strings, default=None
                The filename or list of filename to load.
            target_object : string or list of strings, default='SpikeTrain'
                The type of neo object that should be read out from the input.
                Options are: 'SpikeTrain', 'AnalogSignal'
            kwargs : dict like
                keyword arguments that will be passed to `numpy.loadtxt` see
                https://numpy.org/devdocs/reference/generated/numpy.loadtxt.html
        """
        if target_object not in self.supported_target_objects:
            raise ValueError(f'{target_object} is not a valid object type. '
                             f'Valid values are {self.objects}.')

        # Ensure right dimensionality
        if isinstance(filenames, str):
            filenames = [filenames]

        # Turn kwargs to attributes
        self.filenames = filenames
        self.target_object = target_object

        self.IOs = [ColumnIO(filename, **kwargs) for filename in filenames]

    def __read_analogsignals(
        self,
        gid_list,
        time_unit,
        t_start=None,
        t_stop=None,
        sampling_period=None,
        id_column=0,
        time_column=1,
        value_columns=2,
        value_types=None,
        value_units=None,
    ):
        """
        Internal function called by read_analogsignal() and read_segment().
        """

        # checking gid input parameters
        gid_list, id_column = self._check_input_gids(gid_list, id_column)
        # checking time input parameters
        t_start, t_stop = self._check_input_times(t_start, t_stop, mandatory=False)

        # checking value input parameters
        (value_columns, value_types, value_units) = self._check_input_values_parameters(
            value_columns, value_types, value_units
        )

        # defining standard column order for internal usage
        # [id_column, time_column, value_column1, value_column2, ...]
        column_ids = [id_column, time_column]
        if value_columns is not None:
            column_ids += value_columns
        for i, cid in enumerate(column_ids):
            if cid is None:
                column_ids[i] = -1

        # assert that no single column is assigned twice
        column_list = [id_column, time_column]
        if value_columns is not None:
            column_list += value_columns
        column_list_no_None = [c for c in column_list if c is not None]
        if len(np.unique(column_list_no_None)) < len(column_list_no_None):
            raise ValueError(
                "One or more columns have been specified to contain "
                "the same data. Columns were specified to {column_list_no_None}."
                ""
            )

        # extracting condition and sorting parameters for raw data loading
        (condition, condition_column,
         sorting_column) = self._get_conditions_and_sorting(id_column,
                                                            time_column,
                                                            gid_list,
                                                            t_start,
                                                            t_stop)

        analogsignal_list = []
        for col in self.IOs:

            # loading raw data columns
            data = col.get_columns(
                column_ids=column_ids,
                condition=condition,
                condition_column=condition_column,
                sorting_columns=sorting_column)

            sampling_period = self._check_input_sampling_period(
                                  sampling_period,
                                  time_column,
                                  time_unit,
                                  data)

            # extracting complete gid list for anasig generation
            if (gid_list == []) and id_column is not None:
                current_gid_list = np.unique(data[:, id_column])
            else:
                current_gid_list = gid_list

            # generate analogsignals for each neuron ID
            for i in current_gid_list:
                selected_ids = self._get_selected_ids(
                    i, id_column, time_column, t_start, t_stop, time_unit,
                    data)

                # extract starting time of analogsignal
                if (time_column is not None) and data.size:
                    anasig_start_time = data[selected_ids[0], 1] * time_unit
                else:
                    # set t_start equal to sampling_period because NEST starts
                    #  recording only after 1 sampling_period
                    anasig_start_time = 1. * sampling_period

                if value_columns is not None:
                    # create one analogsignal per value column requested
                    for v_id, value_column in enumerate(value_columns):
                        signal = data[
                            selected_ids[0]:selected_ids[1], value_column]

                        # create AnalogSignal objects and annotate them with
                        #  the neuron ID
                        analogsignal_list.append(AnalogSignal(
                            signal * value_units[v_id],
                            sampling_period=sampling_period,
                            t_start=anasig_start_time,
                            id=i,
                            source_file=col.filename,
                            type=value_types[v_id]))
                        # check for correct length of analogsignal
                        assert (analogsignal_list[-1].t_stop
                                == anasig_start_time + len(signal) *
                                sampling_period)
        return analogsignal_list

    def __read_spiketrains(self, gdf_id_list, time_unit, t_start, t_stop, id_column, time_column, **args):
        """
        Internal function for reading multiple spiketrains at once.
        This function is called by read_spiketrain() and read_segment().
        """
        # assert that the file contains spike times
        if time_column is None:
            raise ValueError("Time column is None. No spike times to " "be read in.")

        gdf_id_list, id_column = self._check_input_gids(gdf_id_list, id_column)

        t_start, t_stop = self._check_input_times(t_start, t_stop, mandatory=True)

        # assert that no single column is assigned twice
        if id_column == time_column:
            raise ValueError("One or more columns have been specified to " "contain the same data.")

        # defining standard column order for internal usage
        # [id_column, time_column, value_column1, value_column2, ...]
        column_ids = [id_column, time_column]
        for i, cid in enumerate(column_ids):
            if cid is None:
                column_ids[i] = -1

        (condition, condition_column, sorting_column) = self._get_conditions_and_sorting(
            id_column, time_column, gdf_id_list, t_start, t_stop
        )

        spiketrain_list = []
        for col in self.IOs:

            data = col.get_columns(
                column_ids=column_ids,
                condition=condition,
                condition_column=condition_column,
                sorting_columns=sorting_column)
            
            # create a list of SpikeTrains for all neuron IDs in gdf_id_list
            # assign spike times to neuron IDs if id_column is given
            if id_column is not None:
                if (gdf_id_list == []) and id_column is not None:
                    current_file_ids = np.unique(data[:, id_column])
                else:
                    current_file_ids = gdf_id_list

                for nid in current_file_ids:
                    selected_ids = self._get_selected_ids(nid, id_column,
                                                          time_column, t_start,
                                                          t_stop, time_unit,
                                                          data)
                    times = data[selected_ids[0]:selected_ids[1], time_column]
                    spiketrain_list.append(SpikeTrain(times, units=time_unit,
                                                      t_start=t_start,
                                                      t_stop=t_stop,
                                                      id=nid,
                                                      source_file=col.filename,
                                                      **args))

            # if id_column is not given, all spike times are collected in one
            #  spike train with id=None
            else:
                train = data[:, time_column]
                spiketrain_list.append([SpikeTrain(train, units=time_unit,
                                                   t_start=t_start,
                                                   t_stop=t_stop,
                                                   id=None,
                                                   source_file=col.filename,
                                                   **args)])
        return spiketrain_list

    def _check_input_times(self, t_start, t_stop, mandatory=True):
        """
        Checks input times for existence and setting default values if
        necessary.

        t_start: pq.quantity.Quantity, start time of the time range to load.
        t_stop: pq.quantity.Quantity, stop time of the time range to load.
        mandatory: bool, if True times can not be None and an error will be
                raised. if False, time values of None will be replaced by
                -infinity or infinity, respectively. default: True.
        """
        if t_stop is None:
            if mandatory:
                raise ValueError("No t_start specified.")
            else:
                t_stop = np.inf * pq.s
        if t_start is None:
            if mandatory:
                raise ValueError("No t_stop specified.")
            else:
                t_start = -np.inf * pq.s

        for time in (t_start, t_stop):
            if not isinstance(time, pq.quantity.Quantity):
                raise TypeError(f"Time value ({time}) is not a quantity.")
        return t_start, t_stop

    def _check_input_values_parameters(self, value_columns, value_types, value_units):
        """
        Checks value parameters for consistency.

        value_columns: int, column id containing the value to load.
        value_types: list of strings, type of values.
        value_units: list of units of the value columns.

        Returns
        adjusted list of [value_columns, value_types, value_units]
        """
        if value_columns is None:
            warnings.warn('No value column was provided.')
            value_types = None
            value_units = None
            return value_columns, value_types, value_units
        if isinstance(value_columns, int):
            value_columns = [value_columns]
        if value_types is None:
            value_types = ["no type"] * len(value_columns)
        elif isinstance(value_types, str):
            value_types = [value_types]

        # translating value types into units as far as possible
        if value_units is None:
            short_value_types = [vtype.split("_")[0] for vtype in value_types]
            if not all([svt in value_type_dict for svt in short_value_types]):
                raise ValueError(f"Can not interpret value types " f'"{value_types}"')
            value_units = [value_type_dict[svt] for svt in short_value_types]

        # checking for same number of value types, units and columns
        if not (len(value_types) == len(value_units) == len(value_columns)):
            raise ValueError(
                "Length of value types, units and columns does "
                f"not match ({len(value_types)},{len(value_units)},{len(value_columns)})"
            )
        if not all([isinstance(vunit, pq.UnitQuantity) for vunit in value_units]):
            raise ValueError("No value unit or standard value type specified.")

        return value_columns, value_types, value_units

    def _check_input_gids(self, gid_list, id_column):
        """
        Checks gid values and column for consistency.

        gid_list: list of int or None, gid to load.
        id_column: int, id of the column containing the gids.

        Returns
        adjusted list of [gid_list, id_column].
        """
        if gid_list is None:
            gid_list = [gid_list]

        if None in gid_list and id_column is not None:
            raise ValueError(
                "No neuron IDs specified but file contains "
                f"neuron IDs in column {str(id_column)}. Specify empty list to "
                "retrieve spiketrains of all neurons."
                ""
            )

        if gid_list != [None] and id_column is None:
            raise ValueError(f"Specified neuron IDs to be {gid_list}, but no ID column " "specified.")
        return gid_list, id_column

    def _check_input_sampling_period(self, sampling_period, time_column, time_unit, data):
        """
        Checks sampling period, times and time unit for consistency.

        sampling_period: pq.quantity.Quantity, sampling period of data to load.
        time_column: int, column id of times in data to load.
        time_unit: pq.quantity.Quantity, unit of time used in the data to load.
        data: numpy array, the data to be loaded / interpreted.

        Returns
        pq.quantities.Quantity object, the updated sampling period.
        """
        if sampling_period is None:
            if time_column is not None:
                data_sampling = np.unique(
                    np.diff(sorted(np.unique(data[:, time_column]))))
                if len(data_sampling) > 1:
                    raise ValueError(f"Different sampling distances found in " "data set ({data_sampling})")
                else:
                    dt = data_sampling[0]
            else:
                raise ValueError('Can not estimate sampling rate without time '
                                 'column id provided.')
            sampling_period = pq.CompoundUnit(str(dt) + '*'
                                              + time_unit.units.u_symbol)
        elif not isinstance(sampling_period, pq.Quantity):
            raise ValueError("sampling_period is not specified as a unit.")
        return sampling_period

    def _get_conditions_and_sorting(self, id_column, time_column, gid_list, t_start, t_stop):
        """
        Calculates the condition, condition_column and sorting_column based on
        other parameters supplied for loading the data.

        id_column: int, id of the column containing gids.
        time_column: int, id of the column containing times.
        gid_list: list of int, gid to be loaded.
        t_start: pq.quantity.Quantity, start of the time range to be loaded.
        t_stop: pq.quantity.Quantity, stop of the time range to be loaded.

        Returns
        updated [condition, condition_column, sorting_column].
        """
        condition, condition_column = None, None
        sorting_column = []
        curr_id = 0
        if (gid_list != [None]) and (gid_list is not None):
            if gid_list != []:

                def condition(x):
                    return x in gid_list

                condition_column = id_column
            sorting_column.append(curr_id)  # Sorting according to gids first
            curr_id += 1
        if time_column is not None:
            sorting_column.append(curr_id)  # Sorting according to time
            curr_id += 1
        elif t_start != -np.inf and t_stop != np.inf:
            warnings.warn("Ignoring t_start and t_stop parameters, because no " "time column id is provided.")
        if sorting_column == []:
            sorting_column = None
        else:
            sorting_column = sorting_column[::-1]
        return condition, condition_column, sorting_column

    def _get_selected_ids(self, gid, id_column, time_column, t_start, t_stop, time_unit, data):
        """
        Calculates the data range to load depending on the selected gid
        and the provided time range (t_start, t_stop)

        gid: int, gid to be loaded.
        id_column: int, id of the column containing gids.
        time_column: int, id of the column containing times.
        t_start: pq.quantity.Quantity, start of the time range to load.
        t_stop: pq.quantity.Quantity, stop of the time range to load.
        time_unit: pq.quantity.Quantity, time unit of the data to load.
        data: numpy array, data to load.

        Returns
        list of selected gids
        """
        gids = np.array([0, data.shape[id_column]])
        if id_column is not None:
            gids = np.array([np.searchsorted(data[:, id_column], gid, side='left'),
                             np.searchsorted(data[:, id_column], gid, side='right')])
        gid_data = data[gids[0]:gids[1], :]

        # select only requested time range
        id_shifts = np.array([0, 0])
        if time_column is not None:
            id_shifts[0] = np.searchsorted(gid_data[:, time_column], 
                                           t_start.rescale(time_unit).magnitude,
                                           side="left")
            id_shifts[1] = (
                np.searchsorted(gid_data[:, time_column], 
                                t_stop.rescale(time_unit).magnitude, 
                                side="left") 
                - gid_data.shape[0]
            )

        selected_ids = gids + id_shifts
        return selected_ids

    def read_block(
        self,
        gid_list=None,
        time_unit=pq.ms,
        t_start=None,
        t_stop=None,
        sampling_period=None,
        id_column_dat=0,
        time_column_dat=1,
        value_columns_dat=2,
        id_column_gdf=0,
        time_column_gdf=1,
        value_types=None,
        value_units=None,
        lazy=False,
    ):
        assert not lazy, "Do not support lazy"

        seg = self.read_segment(gid_list, time_unit, t_start,
                                t_stop, sampling_period, id_column_dat,
                                time_column_dat, value_columns_dat,
                                id_column_gdf, time_column_gdf, value_types,
                                value_units)
        blk = Block(file_origin=seg.file_origin,
                    file_datetime=seg.file_datetime)
        blk.segments.append(seg)
        return blk

    def read_segment(
        self,
        gid_list=None,
        time_unit=pq.ms,
        t_start=None,
        t_stop=None,
        sampling_period=None,
        id_column_dat=0,
        time_column_dat=1,
        value_columns_dat=2,
        id_column_gdf=0,
        time_column_gdf=1,
        value_types=None,
        value_units=None,
        lazy=False,
    ):
        """
        Reads a Segment which contains SpikeTrain(s) with specified neuron IDs
        from the GDF data.

        Arguments
        ----------
        gid_list : list, default: None
            A list of GDF IDs of which to return SpikeTrain(s). gid_list must
            be specified if the GDF file contains neuron IDs, the default None
            then raises an error. Specify an empty list [] to retrieve the
            spike trains of all neurons.
        time_unit : Quantity (time), optional, default: quantities.ms
            The time unit of recorded time stamps in DAT as well as GDF files.
        t_start : Quantity (time), optional, default: 0 * pq.ms
            Start time of SpikeTrain.
        t_stop : Quantity (time), default: None
            Stop time of SpikeTrain. t_stop must be specified, the default None
            raises an error.
        sampling_period : Quantity (frequency), optional, default: None
            Sampling period of the recorded data.
        id_column_dat : int, optional, default: 0
            Column index of neuron IDs in the DAT file.
        time_column_dat : int, optional, default: 1
            Column index of time stamps in the DAT file.
        value_columns_dat : int, optional, default: 2
            Column index of the analog values recorded in the DAT file.
        id_column_gdf : int, optional, default: 0
            Column index of neuron IDs in the GDF file.
        time_column_gdf : int, optional, default: 1
            Column index of time stamps in the GDF file.
        value_types : str, optional, default: None
            Nest data type of the analog values recorded, eg.'V_m', 'I', 'g_e'
        value_units : Quantity (amplitude), default: None
            The physical unit of the recorded signal values.
        lazy : bool, optional, default: False

        Returns
        -------
        seg : Segment
            The Segment contains one SpikeTrain and one AnalogSignal for
            each ID in gid_list.
        """
        assert not lazy, "Do not support lazy"

        if isinstance(gid_list, tuple):
            if gid_list[0] > gid_list[1]:
                raise ValueError("The second entry in gid_list must be " "greater or equal to the first entry.")
            gid_list = range(gid_list[0], gid_list[1] + 1)

        # __read_xxx() needs a list of IDs
        if gid_list is None:
            gid_list = [None]

        # create an empty Segment
        seg = Segment(file_origin=",".join(self.filenames))
        seg.file_datetime = datetime.fromtimestamp(
                                os.stat(self.filenames[-1]).st_mtime)

        # Load analogsignals and attach to Segment
        if 'AnalogSignal' == self.target_object:
            seg.analogsignals = self.__read_analogsignals(
                gid_list,
                time_unit,
                t_start,
                t_stop,
                sampling_period=sampling_period,
                id_column=id_column_dat,
                time_column=time_column_dat,
                value_columns=value_columns_dat,
                value_types=value_types,
                value_units=value_units)
        if 'SpikeTrain' == self.target_object:
            seg.spiketrains = self.__read_spiketrains(
                gid_list, time_unit, t_start, t_stop, id_column=id_column_gdf, time_column=time_column_gdf
            )

        return seg

    def read_analogsignal(
        self,
        gid=None,
        time_unit=pq.ms,
        t_start=None,
        t_stop=None,
        sampling_period=None,
        id_column=0,
        time_column=1,
        value_column=2,
        value_type=None,
        value_unit=None,
        lazy=False,
    ):
        """
        Reads an AnalogSignal with specified neuron ID from the DAT data.

        Arguments
        ----------
        gid : int, default: None
            The GDF ID of the returned SpikeTrain. gdf_id must be specified if
            the GDF file contains neuron IDs, the default None then raises an
            error. Specify an empty list [] to retrieve the spike trains of all
            neurons.
        time_unit : Quantity (time), optional, default: quantities.ms
            The time unit of recorded time stamps.
        t_start : Quantity (time), optional, default: 0 * pq.ms
            Start time of SpikeTrain.
        t_stop : Quantity (time), default: None
            Stop time of SpikeTrain. t_stop must be specified, the default None
            raises an error.
        sampling_period : Quantity (frequency), optional, default: None
            Sampling period of the recorded data.
        id_column : int, optional, default: 0
            Column index of neuron IDs.
        time_column : int, optional, default: 1
            Column index of time stamps.
        value_column : int, optional, default: 2
            Column index of the analog values recorded.
        value_type : str, optional, default: None
            Nest data type of the analog values recorded, eg.'V_m', 'I', 'g_e'.
        value_unit : Quantity (amplitude), default: None
            The physical unit of the recorded signal values.
        lazy : bool, optional, default: False

        Returns
        -------
        spiketrain : SpikeTrain
            The requested SpikeTrain object with an annotation 'id'
            corresponding to the gdf_id parameter.
        """
        assert not lazy, "Do not support lazy"

        # __read_spiketrains() needs a list of IDs
        return self.__read_analogsignals(
            [gid],
            time_unit,
            t_start,
            t_stop,
            sampling_period=sampling_period,
            id_column=id_column,
            time_column=time_column,
            value_columns=value_column,
            value_types=value_type,
            value_units=value_unit,
        )[0]

    def read_spiketrain(
        self, gdf_id=None, time_unit=pq.ms, t_start=None, t_stop=None, id_column=0, time_column=1, lazy=False, **args
    ):
        """
        Reads a SpikeTrain with specified neuron ID from the GDF data.

        Arguments
        ----------
        gdf_id : int, default: None
            The GDF ID of the returned SpikeTrain. gdf_id must be specified if
            the GDF file contains neuron IDs. Providing [] loads all available
            IDs.
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
        lazy : bool, optional, default: False

        Returns
        -------
        spiketrain : SpikeTrain
            The requested SpikeTrain object with an annotation 'id'
            corresponding to the gdf_id parameter.
        """
        assert not lazy, "Do not support lazy"

        if (not isinstance(gdf_id, int)) and gdf_id is not None:
            raise ValueError("gdf_id has to be of type int or None.")

        if gdf_id is None and id_column is not None:
            raise ValueError("No neuron ID specified but file contains " "neuron IDs in column " + str(id_column) + ".")

        return self.__read_spiketrains([gdf_id], time_unit, t_start, t_stop, id_column, time_column, **args)[0]


class ColumnIO:
    """
    Class for reading an ASCII file containing multiple columns of data.
    """

    def __init__(self, filename, **kwargs):
        """
        filename: string, path to ASCII file to read.
        """

        self.filename = filename

        # read the first line to check the data type (int or float) of the data
        f = open(self.filename)
        line = f.readline()
        header_size = 0

        # Check how many header lines the file has so they can be ignored
        while line:
            if line[0].isdigit():
                break
            else:
                header_size += 1
                line = f.readline()

        # Warn user when the header is removed
        if header_size > 0:
            warnings.warn(f'Ignoring {str(header_size)} header lines.')

        if '.' not in line:
            kwargs['dtype'] = np.int32
        else:
            kwargs['dtype'] = np.float32

        self.data = np.loadtxt(self.filename, skiprows=header_size, **kwargs)

        if len(self.data.shape) == 1:
            self.data = self.data[:, np.newaxis]

    def get_columns(self, column_ids="all", condition=None, condition_column=None, sorting_columns=None):
        """
        column_ids : 'all' or list of int, the ids of columns to
                    extract.
        condition : None or function, which is applied to each row to evaluate
                    if it should be included in the result.
                    Needs to return a bool value.
        condition_column : int, id of the column on which the condition
                    function is applied to
        sorting_columns : int or list of int, column ids to sort by.
                    List entries have to be ordered by increasing sorting
                    priority!

        Returns
        -------
        numpy array containing the requested data.
        """

        if column_ids == [] or column_ids == "all":
            column_ids = range(self.data.shape[-1])

        if isinstance(column_ids, (int, float)):
            column_ids = [column_ids]
        column_ids = np.array(column_ids)

        if column_ids is not None:
            if max(column_ids) > len(self.data) - 1:
                raise ValueError('Can not load column ID %i. File contains '
                                 'only %i columns' % (max(column_ids),
                                                      len(self.data)))

        if sorting_columns is not None:
            if isinstance(sorting_columns, int):
                sorting_columns = [sorting_columns]
            if max(sorting_columns) >= self.data.shape[1]:
                raise ValueError(
                    f"Can not sort by column ID {max(sorting_columns)}. File contains "
                    f"only {self.data.shape[1]} columns"
                )

        # Starting with whole dataset being selected for return
        selected_data = self.data

        # Apply filter condition to rows
        if condition and (condition_column is None):
            raise ValueError("Filter condition provided, but no " "condition_column ID provided")
        elif (condition_column is not None) and (condition is None):
            warnings.warn("Condition column ID provided, but no condition " "given. No filtering will be performed.")

        elif (condition is not None) and (condition_column is not None):
            condition_function = np.vectorize(condition)
            mask = condition_function(selected_data[:, condition_column]).astype(bool)
            selected_data = selected_data[mask, :]

        # Apply sorting if requested
        if sorting_columns is not None:
            values_to_sort = selected_data[:, sorting_columns].T
            ordered_ids = np.lexsort(tuple(values_to_sort[i] for i in range(len(values_to_sort))))
            selected_data = selected_data[ordered_ids, :]

        # Select only requested columns
        selected_data = selected_data[:, column_ids]

        return selected_data
