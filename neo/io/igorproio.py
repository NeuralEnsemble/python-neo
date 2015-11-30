# -*- coding: utf-8 -*-
"""
Class for reading data created by IGOR Pro (WaveMetrics, Inc., Portland, OR, USA)

Depends on: igor (https://pypi.python.org/pypi/igor/)

Supported: Read

Author: Andrew Davison

"""

from __future__ import absolute_import
from datetime import datetime
from warnings import warn
import numpy as np
import quantities as pq
from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal
import igor.binarywave as bw


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
        block.segments.append(self.read_segment(lazy=lazy, cascade=cascade))
        return block

    def read_segment(self, lazy=False, cascade=True):
        segment = Segment(file_origin=self.filename)
        segment.analogsignals.append(self.read_analogsignal(lazy=lazy, cascade=cascade))
        return segment

    def read_analogsignal(self, lazy=False, cascade=True):
        content = bw.load(self.filename)['wave']
        assert content['padding'].size == 0, "Cannot handle non-empty padding"
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

        return AnalogSignal(signal, units=units, copy=False, t_start=t_start,
                            sampling_period=sampling_period, name=name,
                            file_origin=self.filename, **annotations)


def key_value_string_parser(itemsep=";", kvsep=":"):
    """
    Parses a string into a dict.

    Arguments:
        itemsep - character which separates items
        kvsep - character which separates the key and value within an item

    Returns:
        a function which takes the string to be parses as the sole argument and returns a dict.

    Example:

        >>> parse = key_value_string_parser(itemsep=";", kvsep=":")
        >>> parse("a:2;b:3")
        {'a': 2, 'b': 3}
    """
    def parser(s):
        items = s.split(itemsep)
        return dict(item.split(kvsep, 1) for item in items if item)
    return parser
