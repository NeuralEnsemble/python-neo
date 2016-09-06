# -*- coding: utf-8 -*-
"""
Class for reading data created by IGOR Pro (WaveMetrics, Inc., Portland, OR, USA)

Depends on: igor (https://pypi.python.org/pypi/igor/)

Supported: Read

Author: Andrew Davison

"""

from __future__ import absolute_import
from warnings import warn
import numpy as np
import quantities as pq
from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal
try:
    import igor.binarywave as bw
    HAVE_IGOR = True
except ImportError:
    HAVE_IGOR = False


class IgorIO(BaseIO):
    """
    Class for reading Igor Binary Waves (.ibw) written by WaveMetrics’ IGOR Pro software.

    Support for Packed Experiment (.pxp) files is planned.

    It requires the `igor` Python package by W. Trevor King.

    Usage:
        >>> from neo import io
        >>> r = io.IgorIO(filename='...ibw')



    """

    is_readable = True   # This class can only read data
    is_writable = False  # write is not supported
    supported_objects = [Block, Segment, AnalogSignal]
    readable_objects = [Block, Segment , AnalogSignal]
    writeable_objects = []
    has_header = False
    is_streameable = False
    name = 'igorpro'
    extensions = ['ibw'] #, 'pxp']
    mode = 'file'

    def __init__(self, filename=None, parse_notes=None) :
        """


        Arguments:
            filename: the filename

            parse_notes: (optional) A function which will parse the 'notes'
            field in the file header and return a dictionary which will be
            added to the object annotations.

        """
        BaseIO.__init__(self)
        self.filename = filename
        self.parse_notes = parse_notes

    def read_block(self, lazy=False, cascade=True):
        block = Block(file_origin=self.filename)
        if cascade:
            block.segments.append(self.read_segment(lazy=lazy, cascade=cascade))
            block.segments[-1].block = block
        return block

    def read_segment(self, lazy=False, cascade=True):
        segment = Segment(file_origin=self.filename)
        if cascade:
            segment.analogsignals.append(self.read_analogsignal(lazy=lazy, cascade=cascade))
            segment.analogsignals[-1].segment = segment
        return segment

    def read_analogsignal(self, lazy=False, cascade=True):
        if not HAVE_IGOR:
            raise Exception("igor package not installed. Try `pip install igor`")
        data = bw.load(self.filename)
        version = data['version']
        if version > 3:
            raise IOError("Igor binary wave file format version {0} is not supported.".format(version))
        content = data['wave']
        if "padding" in content:
            assert content['padding'].size == 0, "Cannot handle non-empty padding"
        if lazy:
            # not really lazy, since the `igor` module loads the data anyway
            signal = np.array((), dtype=content['wData'].dtype)
        else:
            signal = content['wData']
        note = content['note']
        header = content['wave_header']
        name = header['bname']
        assert header['botFullScale'] == 0
        assert header['topFullScale'] == 0
        units = "".join(header['dataUnits'])
        time_units = "".join(header['xUnits']) or "s"
        t_start = pq.Quantity(header['hsB'], time_units)
        sampling_period = pq.Quantity(header['hsA'], time_units)
        if self.parse_notes:
            try:
                annotations = self.parse_notes(note)
            except ValueError:
                warn("Couldn't parse notes field.")
                annotations = {'note': note}
        else:
            annotations = {'note': note}

        signal = AnalogSignal(signal, units=units, copy=False, t_start=t_start,
                              sampling_period=sampling_period, name=name,
                              file_origin=self.filename, **annotations)
        if lazy:
            signal.lazy_shape = content['wData'].shape
        return signal


# the following function is to handle the annotations in the
# Igor data files from the Blue Brain Project NMC Portal
def key_value_string_parser(itemsep=";", kvsep=":"):
    """
    Parses a string into a dict.

    Arguments:
        itemsep - character which separates items
        kvsep - character which separates the key and value within an item

    Returns:
        a function which takes the string to be parsed as the sole argument and returns a dict.

    Example:

        >>> parse = key_value_string_parser(itemsep=";", kvsep=":")
        >>> parse("a:2;b:3")
        {'a': 2, 'b': 3}
    """
    def parser(s):
        items = s.split(itemsep)
        return dict(item.split(kvsep, 1) for item in items if item)
    return parser
