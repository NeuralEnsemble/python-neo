# -*- coding: utf-8 -*-
"""
Class for reading/writing analog signals in a text file.
Each column represents an AnalogSignal. All AnalogSignals have the same sampling rate.
Covers many cases when parts of a file can be viewed as a CSV format.

Supported : Read/Write

Author: sgarcia

"""

import csv
import os

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import AnalogSignal, Segment, Block


class AsciiSignalIO(BaseIO):
    """

    Class for reading signals in generic ascii format.
    Columns represent signals. They all share the same sampling rate.
    The sampling rate is externally known or the first column could hold the time vector.

    Usage:
        >>> from neo import io
        >>> r = io.AsciiSignalIO(filename='File_asciisignal_2.txt')
        >>> seg = r.read_segment()
        >>> print seg.analogsignals
        [<AnalogSignal(array([ 39.0625    ,   0.        ,   0.        , ..., -26.85546875 ...

    Arguments relevant for reading and writing:
        delimiter : column delimiter in file, e.g. '\t', one space, two spaces, ',', ';'
        timecolumn :  None or a valid integer that identifies which column contains
                      the time vector (counting from zero)
        units : units of AnalogSignal can be a str or directly a Quantity
        time_units : where timecolumn is specified, the time units must be specified
                     as a string or Quantity

    Arguments relevant only for reading:
        usecols : if None take all columns otherwise a list for selected columns
                  (counting from zero)
        skiprows : skip n first lines in case they contains header informations
        sampling_rate : the sampling rate of signals. Ignored if timecolumn is not None
        t_start : time of the first sample (Quantity). Ignored if timecolumn is not None
        signal_group_mode : if 'all-in-one', load data as a single, multi-channel AnalogSignal,
                       if 'split-all' (default for backwards compatibility) load data as
                       separate, single-channel AnalogSignals
        method : 'genfromtxt', 'csv', 'homemade' or a user-defined function which takes a
                 filename and usecolumns as argument and returns a 2D NumPy array.

        If specifying both usecols and timecolumn, the latter should identify
        the column index _after_ removing the unused columns.

        The methods are as follows:
            - 'genfromtxt' use numpy.genfromtxt
            - 'csv' use csv module
            - 'homemade' use an intuitive, more robust but slow method

    """

    is_readable = True
    is_writable = True

    supported_objects = [Block, Segment, AnalogSignal]
    readable_objects = [Block, Segment]
    # can write a Block with a single segment, but not the general case
    writeable_objects = [Segment]

    has_header = False
    is_streameable = False

    read_params = {
        Segment: [
            ('delimiter', {'value': '\t', 'possible': ['\t', ' ', ',', ';']}),
            ('usecols', {'value': None, 'type': int}),
            ('skiprows', {'value': 0}),
            ('timecolumn', {'value': None, 'type': int}),
            ('units', {'value': 'V', }),
            ('time_units', {'value': pq.s, }),
            ('sampling_rate', {'value': 1.0 * pq.Hz, }),
            ('t_start', {'value': 0.0 * pq.s, }),
            ('method', {'value': 'homemade', 'possible': ['genfromtxt', 'csv', 'homemade']}),
            ('signal_group_mode', {'value': 'split-all'})
        ]
    }
    write_params = {
        Segment: [
            ('delimiter', {'value': '\t', 'possible': ['\t', ' ', ',', ';']}),
            ('writetimecolumn', {'value': True, }),
        ]
    }

    name = None
    extensions = ['txt', 'asc', ]

    mode = 'file'

    def __init__(self, filename=None, delimiter='\t', usecols=None, skiprows=0, timecolumn=None,
                 sampling_rate=1.0 * pq.Hz, t_start=0.0 * pq.s, units=pq.V, time_units=pq.s,
                 method='genfromtxt', signal_group_mode='split-all'):
        """
        This class read/write AnalogSignal in a text file.
        Each signal is a column.
        One of the columns can be the time vector.

        Arguments:
            filename : the filename to read/write
        """
        # todo: allow units to be a list/array (e.g. current and voltage in the same file)
        BaseIO.__init__(self)
        self.filename = filename
        self.delimiter = delimiter
        self.usecols = usecols
        self.skiprows = skiprows
        self.timecolumn = timecolumn
        self.sampling_rate = sampling_rate
        self.time_units = time_units
        if time_units is not None:
            self.time_units = pq.Quantity(1, time_units)
        if isinstance(t_start, pq.Quantity):
            self.t_start = t_start
        else:
            if not isinstance(self.time_units, pq.Quantity):
                raise ValueError("Units of t_start not specified")
            self.t_start = t_start * self.time_units
        self.units = pq.Quantity(1, units)

        if not(method in ('genfromtxt', 'csv', 'homemade') or callable(method)):
            raise ValueError(
                "method must be one of 'genfromtxt', 'csv', 'homemade', or a function")
        self.method = method
        self.signal_group_mode = signal_group_mode

    def read_block(self, lazy=False):
        block = Block(file_origin=os.path.basename(self.filename))
        segment = self.read_segment(lazy=lazy)
        segment.block = block
        block.segments.append(segment)
        return block

    def read_segment(self, lazy=False):
        """

        """
        if lazy:
            raise NotImplementedError("lazy mode not supported")

        seg = Segment(file_origin=os.path.basename(self.filename))

        # loadtxt
        if self.method == 'genfromtxt':
            sig = np.genfromtxt(self.filename,
                                delimiter=self.delimiter,
                                usecols=self.usecols,
                                skip_header=self.skiprows,
                                dtype='f')
            if len(sig.shape) == 1:
                sig = sig[:, np.newaxis]
        elif self.method == 'csv':
            tab = [l for l in csv.reader(file(self.filename, 'rU'), delimiter=self.delimiter)]
            tab = tab[self.skiprows:]
            sig = np.array(tab, dtype='f')
            if self.usecols is not None:
                mask = np.array(self.usecols)
                sig = sig[:, mask]
        elif self.method == 'homemade':
            fid = open(self.filename, 'rU')
            for l in range(self.skiprows):
                fid.readline()
            tab = []
            for line in fid.readlines():
                line = line.replace('\r', '')
                line = line.replace('\n', '')
                parts = line.split(self.delimiter)
                while '' in parts:
                    parts.remove('')
                tab.append(parts)
            sig = np.array(tab, dtype='f')
            if self.usecols is not None:
                mask = np.array(self.usecols)
                sig = sig[:, mask]
        else:
            sig = self.method(self.filename, self.usecols)
            if not isinstance(sig, np.ndarray):
                raise TypeError("method function must return a NumPy array")
            if len(sig.shape) == 1:
                sig = sig[:, np.newaxis]
            elif sig.shape != 2:
                raise ValueError("method function must return a 1D or 2D NumPy array")

        if self.timecolumn is None:
            sampling_rate = self.sampling_rate
            t_start = self.t_start
        else:
            # todo: if the values in timecolumn are not equally spaced
            #       (within float representation tolerances)
            #       we should produce an IrregularlySampledSignal
            sampling_rate = 1.0 / np.mean(np.diff(sig[:, self.timecolumn])) / self.time_units
            t_start = sig[0, self.timecolumn] * self.time_units

        if self.signal_group_mode == 'all-in-one':
            if self.timecolumn is not None:
                mask = list(range(sig.shape[1]))
                if self.timecolumn >= 0:
                    mask.remove(self.timecolumn)
                else:  # allow negative column index
                    mask.remove(sig.shape[1] + self.timecolumn)
                signal = sig[:, mask]
            else:
                signal = sig
            anaSig = AnalogSignal(signal * self.units, sampling_rate=sampling_rate,
                                  t_start=t_start,
                                  channel_index=self.usecols or np.arange(signal.shape[1]),
                                  name='multichannel')
            seg.analogsignals.append(anaSig)
        else:
            if self.timecolumn is not None and self.timecolumn < 0:
                time_col = sig.shape[1] + self.timecolumn
            else:
                time_col = self.timecolumn
            for i in range(sig.shape[1]):
                if time_col == i:
                    continue
                signal = sig[:, i] * self.units
                anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                      t_start=t_start, channel_index=i,
                                      name='Column %d' % i)
                seg.analogsignals.append(anaSig)

        seg.create_many_to_one_relationship()
        return seg

    def write_segment(self, segment):
        """
        Write a segment and AnalogSignal in a text file.
        """
        # todo: check all analog signals have the same length, physical dimensions
        # and sampling rates
        l = []
        if self.timecolumn is not None:
            if self.timecolumn != 0:
                raise NotImplementedError("Only column 0 currently supported for writing times")
            l.append(segment.analogsignals[0].times[:, np.newaxis].rescale(self.time_units))
        # todo: check signals are compatible (size, sampling rate), otherwise we
        # can't/shouldn't concatenate them
        for anaSig in segment.analogsignals:
            l.append(anaSig.rescale(self.units).magnitude)
        sigs = np.concatenate(l, axis=1)
        # print sigs.shape
        np.savetxt(self.filename, sigs, delimiter=self.delimiter)

    def write_block(self, block):
        """
        Can only write blocks containing a single segment.
        """
        # in future, maybe separate segments by a blank link, or a "magic" comment
        if len(block.segments) > 1:
            raise ValueError("Can only write blocks containing a single segment."
                             " This block contains {} segments.".format(len(block.segments)))
        self.write_segment(block.segments[0])
