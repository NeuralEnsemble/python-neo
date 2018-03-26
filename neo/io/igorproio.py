# -*- coding: utf-8 -*-
"""
Class for reading data created by IGOR Pro
(WaveMetrics, Inc., Portland, OR, USA)

Depends on: igor (https://pypi.python.org/pypi/igor/)

Supported: Read

Author: Andrew Davison
Also contributing: Rick Gerkin

"""

from __future__ import absolute_import
from warnings import warn
import numpy as np
import quantities as pq
from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal

try:
    import igor.binarywave as bw
    import igor.packed as pxp

    HAVE_IGOR = True
except ImportError:
    HAVE_IGOR = False


class IgorIO(BaseIO):
    """
    Class for reading Igor Binary Waves (.ibw)
    or Packed Experiment (.pxp) files written by WaveMetrics’
    IGOR Pro software.

    It requires the `igor` Python package by W. Trevor King.

    Usage:
        >>> from neo import io
        >>> r = io.IgorIO(filename='...ibw')



    """

    is_readable = True  # This class can only read data
    is_writable = False  # write is not supported
    supported_objects = [Block, Segment, AnalogSignal]
    readable_objects = [Block, Segment, AnalogSignal]
    writeable_objects = []
    has_header = False
    is_streameable = False
    name = 'igorpro'
    extensions = ['ibw', 'pxp']
    mode = 'file'

    def __init__(self, filename=None, parse_notes=None):
        """


        Arguments:
            filename: the filename

            parse_notes: (optional) A function which will parse the 'notes'
            field in the file header and return a dictionary which will be
            added to the object annotations.

        """
        BaseIO.__init__(self)
        assert any([filename.endswith('.%s' % x) for x in self.extensions]), \
            "Only the following extensions are supported: %s" % self.extensions
        self.filename = filename
        self.extension = filename.split('.')[-1]
        self.parse_notes = parse_notes

    def read_block(self, lazy=False):
        assert not lazy, 'Do not support lazy'

        block = Block(file_origin=self.filename)
        block.segments.append(self.read_segment(lazy=lazy))
        block.segments[-1].block = block
        return block

    def read_segment(self, lazy=False):
        assert not lazy, 'Do not support lazy'

        segment = Segment(file_origin=self.filename)
        segment.analogsignals.append(
            self.read_analogsignal(lazy=lazy))
        segment.analogsignals[-1].segment = segment
        return segment

    def read_analogsignal(self, path=None, lazy=False):
        assert not lazy, 'Do not support lazy'

        if not HAVE_IGOR:
            raise Exception(("`igor` package not installed. "
                             "Try `pip install igor`"))
        if self.extension == 'ibw':
            data = bw.load(self.filename)
            version = data['version']
            if version > 5:
                raise IOError(("Igor binary wave file format version {0} "
                               "is not supported.".format(version)))
        elif self.extension == 'pxp':
            assert type(path) is str, \
                "A colon-separated Igor-style path must be provided."
            _, filesystem = pxp.load(self.filename)
            path = path.split(':')
            location = filesystem['root']
            for element in path:
                if element != 'root':
                    location = location[element.encode('utf8')]
            data = location.wave
        content = data['wave']
        if "padding" in content:
            assert content['padding'].size == 0, \
                "Cannot handle non-empty padding"
        signal = content['wData']
        note = content['note']
        header = content['wave_header']
        name = str(header['bname'].decode('utf-8'))
        units = "".join([x.decode() for x in header['dataUnits']])
        try:
            time_units = "".join([x.decode() for x in header['xUnits']])
            assert len(time_units)
        except:
            time_units = "s"
        try:
            t_start = pq.Quantity(header['hsB'], time_units)
        except KeyError:
            t_start = pq.Quantity(header['sfB'][0], time_units)
        try:
            sampling_period = pq.Quantity(header['hsA'], time_units)
        except:
            sampling_period = pq.Quantity(header['sfA'][0], time_units)
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
        a function which takes the string to be parsed as the sole argument
        and returns a dict.

    Example:

        >>> parse = key_value_string_parser(itemsep=";", kvsep=":")
        >>> parse("a:2;b:3")
        {'a': 2, 'b': 3}
    """

    def parser(s):
        items = s.split(itemsep)
        return dict(item.split(kvsep, 1) for item in items if item)

    return parser
