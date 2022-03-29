"""
Class for reading/writing analog signals in a text file.
Each column represents an AnalogSignal. All AnalogSignals have the same sampling rate.
Covers many cases when parts of a file can be viewed as a CSV format.

Supported : Read/Write

Author: sgarcia

"""

import csv
import os
import json

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import AnalogSignal, IrregularlySampledSignal, Segment, Block


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
        delimiter:
            column delimiter in file, e.g. '\t', one space, two spaces, ',', ';'
        timecolumn:
            None or a valid integer that identifies which column contains the time vector
            (counting from zero)
        units:
            units of AnalogSignal can be a str or directly a Quantity
        time_units:
            where timecolumn is specified, the time units must be specified as a string or
            Quantity
        metadata_filename:
            the path to a JSON file containing metadata

    Arguments relevant only for reading:
        usecols:
            if None take all columns otherwise a list for selected columns (counting from zero)
        skiprows:
            skip n first lines in case they contains header informations
        sampling_rate:
            the sampling rate of signals. Ignored if timecolumn is not None
        t_start:
            time of the first sample (Quantity). Ignored if timecolumn is not None
        signal_group_mode:
            if 'all-in-one', load data as a single, multi-channel AnalogSignal, if 'split-all'
            (default for backwards compatibility) load data as separate, single-channel
            AnalogSignals
        method:
            'genfromtxt', 'csv', 'homemade' or a user-defined function which takes a filename and
            usecolumns as argument and returns a 2D NumPy array.

    If specifying both usecols and timecolumn, the latter should identify
    the column index _after_ removing the unused columns.

    The methods are as follows:
        - 'genfromtxt' use numpy.genfromtxt
        - 'csv' use csv module
        - 'homemade' use an intuitive, more robust but slow method

    If `metadata_filename` is provided, the parameters for reading/writing the file
    ("delimiter", "timecolumn", "units", etc.) will be read from that file.
    IF a metadata filename is not provided, the IO will look for a JSON file in the same
    directory with a matching filename, e.g. if the datafile was named "foo.txt" then the
    IO would automatically look for a file called "foo_about.json"
    If parameters are specified both in the metadata file and as arguments to the IO constructor,
    the former will take precedence.

    Example metadata file::

        {
            "filename": "foo.txt",
            "delimiter": " ",
            "timecolumn": 0,
            "units": "pA",
            "time_units": "ms",
            "sampling_rate": {
                "value": 1.0,
                "units": "kHz"
            },
            "method": "genfromtxt",
            "signal_group_mode": 'all-in-one'
        }
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
    extensions = ['txt', 'asc', 'csv', 'tsv']

    mode = 'file'

    def __init__(self, filename=None, delimiter='\t', usecols=None, skiprows=0, timecolumn=None,
                 sampling_rate=1.0 * pq.Hz, t_start=0.0 * pq.s, units=pq.V, time_units=pq.s,
                 method='genfromtxt', signal_group_mode='split-all', metadata_filename=None):
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
        self.metadata_filename = metadata_filename
        metadata = self.read_metadata()
        self.delimiter = metadata.get("delimiter", delimiter)
        self.usecols = metadata.get("usecols", usecols)
        self.skiprows = metadata.get("skiprows", skiprows)
        self.timecolumn = metadata.get("timecolumn", timecolumn)
        self.sampling_rate = metadata.get("sampling_rate", sampling_rate)
        self.time_units = metadata.get("time_units", time_units)
        if self.time_units is not None:
            self.time_units = pq.Quantity(1, self.time_units)
        self.t_start = metadata.get("t_start", t_start)
        if not isinstance(t_start, pq.Quantity):
            if not isinstance(self.time_units, pq.Quantity):
                raise ValueError("Units of t_start not specified")
            self.t_start *= self.time_units
        self.units = metadata.get("units", pq.Quantity(1, units))

        self.method = metadata.get("method", method)
        if not(self.method in ('genfromtxt', 'csv', 'homemade') or callable(self.method)):
            raise ValueError(
                "method must be one of 'genfromtxt', 'csv', 'homemade', or a function")

        self.signal_group_mode = metadata.get("signal_group_mode", signal_group_mode)

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
                                dtype='f',
                                filling_values=None,
                                comments='""',
                                names=None,
                                loose=True,
                                invalid_raise=False)
            if len(sig.shape) == 1:
                sig = sig[:, np.newaxis]
        elif self.method == 'csv':
            with open(self.filename, newline=None) as fp:
                tab = [l for l in csv.reader(fp, delimiter=self.delimiter)]
            tab = tab[self.skiprows:]
            sig = np.array(tab, dtype='f')
            if self.usecols is not None:
                mask = np.array(self.usecols)
                sig = sig[:, mask]
        elif self.method == 'homemade':
            with open(self.filename, 'r', newline=None) as fid:
                for _ in range(self.skiprows):
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
            elif len(sig.shape) != 2:
                raise ValueError("method function must return a 1D or 2D NumPy array")

        if self.timecolumn is None:
            sampling_rate = self.sampling_rate
            t_start = self.t_start
        else:
            delta_t = np.diff(sig[:, self.timecolumn])
            mean_delta_t = np.mean(delta_t)
            if (delta_t.max() - delta_t.min()) / mean_delta_t < 1e-6:
                # equally spaced --> AnalogSignal
                sampling_rate = 1.0 / np.mean(np.diff(sig[:, self.timecolumn])) / self.time_units
            else:
                # not equally spaced --> IrregularlySampledSignal
                sampling_rate = None
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
            if sampling_rate is None:
                irr_sig = IrregularlySampledSignal(signal[:, self.timecolumn] * self.time_units,
                                                   signal * self.units,
                                                   name='multichannel')
                seg.irregularlysampledsignals.append(irr_sig)
            else:
                ana_sig = AnalogSignal(signal * self.units, sampling_rate=sampling_rate,
                                       t_start=t_start,
                                       channel_index=self.usecols or np.arange(signal.shape[1]),
                                       name='multichannel')
                seg.analogsignals.append(ana_sig)
        else:
            if self.timecolumn is not None and self.timecolumn < 0:
                time_col = sig.shape[1] + self.timecolumn
            else:
                time_col = self.timecolumn
            for i in range(sig.shape[1]):
                if time_col == i:
                    continue
                signal = sig[:, i] * self.units
                if sampling_rate is None:
                    irr_sig = IrregularlySampledSignal(sig[:, time_col] * self.time_units,
                                                       signal,
                                                       t_start=t_start, channel_index=i,
                                                       name='Column %d' % i)
                    seg.irregularlysampledsignals.append(irr_sig)
                else:
                    ana_sig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                           t_start=t_start, channel_index=i,
                                           name='Column %d' % i)
                    seg.analogsignals.append(ana_sig)

        seg.create_many_to_one_relationship()
        return seg

    def read_metadata(self):
        """
        Read IO parameters from an associated JSON file
        """
        # todo: also read annotations
        if self.metadata_filename is None:
            candidate = os.path.splitext(self.filename)[0] + "_about.json"
            if os.path.exists(candidate):
                self.metadata_filename = candidate
            else:
                return {}
        if os.path.exists(self.metadata_filename):
            with open(self.metadata_filename) as fp:
                metadata = json.load(fp)
            for key in "sampling_rate", "t_start":
                if key in metadata:
                    metadata[key] = pq.Quantity(metadata[key]["value"], metadata[key]["units"])
            for key in "units", "time_units":
                if key in metadata:
                    metadata[key] = pq.Quantity(1, metadata[key])
            return metadata
        else:
            return {}

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
        # check signals are compatible (size, sampling rate), otherwise we
        # can't/shouldn't concatenate them
        # also set sampling_rate, t_start, units, time_units from signal(s)
        signal0 = segment.analogsignals[0]
        for attr in ("sampling_rate", "units", "shape"):
            val0 = getattr(signal0, attr)
            for signal in segment.analogsignals[1:]:
                val1 = getattr(signal, attr)
                if val0 != val1:
                    raise Exception("Signals being written have different " + attr)
            setattr(self, attr, val0)
        # todo t_start, time_units
        self.time_units = signal0.times.units
        self.t_start = min(sig.t_start for sig in segment.analogsignals)

        for anaSig in segment.analogsignals:
            l.append(anaSig.rescale(self.units).magnitude)
        sigs = np.concatenate(l, axis=1)
        # print sigs.shape
        np.savetxt(self.filename, sigs, delimiter=self.delimiter)
        if self.metadata_filename is not None:
            self.write_metadata()

    def write_block(self, block):
        """
        Can only write blocks containing a single segment.
        """
        # in future, maybe separate segments by a blank link, or a "magic" comment
        if len(block.segments) > 1:
            raise ValueError("Can only write blocks containing a single segment."
                             " This block contains {} segments.".format(len(block.segments)))
        self.write_segment(block.segments[0])

    def write_metadata(self, metadata_filename=None):
        """
        Write IO parameters to an associated JSON file
        """
        # todo: also write annotations
        metadata = {
            "filename": self.filename,
            "delimiter": self.delimiter,
            "usecols": self.usecols,
            "skiprows": self.skiprows,
            "timecolumn": self.timecolumn,
            "sampling_rate": {
                "value": float(self.sampling_rate.magnitude),
                "units": self.sampling_rate.dimensionality.string
            },
            "t_start": {
                "value": float(self.t_start.magnitude),
                "units": self.t_start.dimensionality.string
            },
            "units": self.units.dimensionality.string,
            "time_units": self.time_units.dimensionality.string,
            "method": self.method,
            "signal_group_mode": self.signal_group_mode
        }
        if metadata_filename is None:
            if self.metadata_filename is None:
                self.metadata_filename = os.path.splitext(self.filename) + "_about.json"
        else:
            self.metadata_filename = metadata_filename
        with open(self.metadata_filename, "w") as fp:
            json.dump(metadata, fp)
        return self.metadata_filename
