"""
Class for reading data created by IGOR Pro
(WaveMetrics, Inc., Portland, OR, USA)

Depends on: igor (https://pypi.python.org/pypi/igor/)

Supported: Read

Author: Andrew Davison
Also contributing: Rick Gerkin

"""

from warnings import warn
import pathlib
import quantities as pq
from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal



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
        filename = pathlib.Path(filename)
        assert filename.suffix[1:] in self.extensions, \
            "Only the following extensions are supported: %s" % self.extensions
        self.filename = filename
        self.extension = filename.suffix[1:]
        self.parse_notes = parse_notes
        self._filesystem = None

    def read_block(self, lazy=False):
        assert not lazy, 'This IO does not support lazy mode'

        block = Block(file_origin=str(self.filename))
        block.segments.append(self.read_segment(lazy=lazy))
        block.segments[-1].block = block
        return block

    def read_segment(self, lazy=False):
        import igor.packed as pxp
        from igor.record.wave import WaveRecord

        assert not lazy, 'This IO does not support lazy mode'
        segment = Segment(file_origin=str(self.filename))

        if self.extension == 'pxp':
            if not self._filesystem:
                _, self.filesystem = pxp.load(str(self.filename))

            def callback(dirpath, key, value):
                if isinstance(value, WaveRecord):
                    signal = self._wave_to_analogsignal(value.wave['wave'], dirpath)
                    signal.segment = segment
                    segment.analogsignals.append(signal)

            pxp.walk(self.filesystem, callback)
        else:
            segment.analogsignals.append(
                self.read_analogsignal(lazy=lazy))
            segment.analogsignals[-1].segment = segment
        return segment

    def read_analogsignal(self, path=None, lazy=False):
        import igor.binarywave as bw
        import igor.packed as pxp

        assert not lazy, 'This IO does not support lazy mode'

        if self.extension == 'ibw':
            data = bw.load(str(self.filename))
            version = data['version']
            if version > 5:
                raise IOError("Igor binary wave file format version {} "
                               "is not supported.".format(version))
        elif self.extension == 'pxp':
            assert type(path) is str, \
                "A colon-separated Igor-style path must be provided."
            if not self._filesystem:
                _, self.filesystem = pxp.load(str(self.filename))
                path = path.split(':')
                location = self.filesystem['root']
                for element in path:
                    if element != 'root':
                        location = location[element.encode('utf8')]
            data = location.wave

        return self._wave_to_analogsignal(data['wave'], [])

    def _wave_to_analogsignal(self, content, dirpath):
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
        annotations["igor_path"] = ":".join(item.decode('utf-8') for item in dirpath)

        signal = AnalogSignal(signal, units=units, copy=False, t_start=t_start,
                              sampling_period=sampling_period, name=name,
                              file_origin=str(self.filename), **annotations)
        return signal


# the following function is to handle the annotations in the
# Igor data files from the Blue Brain Project NMC Portal
def key_value_string_parser(itemsep=";", kvsep=":"):
    """
    Parses a string into a dict.

    Parameters
    ----------
    itemsep : str
        Character which separates items
    kvsep : str
        Character which separates the key and value within an item

    Returns
    -------
    callable
        a function which takes the string to be parsed as the sole argument
        and returns a dict.

    Examples
    --------

        >>> parse = key_value_string_parser(itemsep=";", kvsep=":")
        >>> parse("a:2;b:3")
        {'a': 2, 'b': 3}
    """

    def parser(s):
        items = s.split(itemsep)
        return dict(item.split(kvsep, 1) for item in items if item)

    return parser
