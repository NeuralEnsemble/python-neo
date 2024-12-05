"""
Class for reading data created by IGOR Pro
(WaveMetrics, Inc., Portland, OR, USA)

Depends on: igor2 (https://pypi.python.org/pypi/igor2/)

Supported: Read

Author: Andrew Davison
Also contributing: Rick Gerkin

"""

from warnings import warn
import pathlib

import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal, NeoReadWriteError


class IgorIO(BaseIO):
    """
    Class for reading Igor Binary Waves (.ibw)
    or Packed Experiment (.pxp) files written by WaveMetrics’
    IGOR Pro software.

    It requires the `igor2` Python package.

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
    name = "igorpro"
    extensions = ["ibw", "pxp"]
    mode = "file"

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
        if filename.suffix[1:] not in self.extensions:
            raise NotImplementedError(f"Only the following extensions are supported: {self.extensions}")
        self.filename = filename
        self.extension = filename.suffix[1:]
        self.parse_notes = parse_notes
        self._filesystem = None

    def read_block(self, lazy=False):
        if lazy:
            raise NeoReadWriteError("This IO does not support lazy reading")

        block = Block(file_origin=str(self.filename))
        block.segments.append(self.read_segment(lazy=lazy))
        return block

    def read_segment(self, lazy=False):
        import igor2.packed as pxp
        from igor2.record.wave import WaveRecord

        if lazy:
            raise NeoReadWriteError("This IO does not support lazy mode")
        segment = Segment(file_origin=str(self.filename))

        if self.extension == "pxp":
            if not self._filesystem:
                _, self.filesystem = pxp.load(str(self.filename))

            def callback(dirpath, key, value):
                if isinstance(value, WaveRecord):
                    signal = self._wave_to_analogsignal(value.wave["wave"], dirpath)
                    segment.analogsignals.append(signal)

            pxp.walk(self.filesystem, callback)
        else:
            segment.analogsignals.append(self.read_analogsignal(lazy=lazy))
        return segment

    def read_analogsignal(self, path=None, lazy=False):
        import igor2.binarywave as bw
        import igor2.packed as pxp

        if lazy:
            raise NeoReadWriteError("This IO does not support lazy mode")

        if self.extension == "ibw":
            data = bw.load(str(self.filename))
            version = data["version"]
            if version > 5:
                raise IOError(f"Igor binary wave file format version {version} " "is not supported.")
        elif self.extension == "pxp":
            if type(path) is not str:
                raise TypeError("A colon-separated Igor-style path must be provided.")
            if not self._filesystem:
                _, self.filesystem = pxp.load(str(self.filename))
                path = path.split(":")
                location = self.filesystem["root"]
                for element in path:
                    if element != "root":
                        location = location[element.encode("utf8")]
            data = location.wave

        return self._wave_to_analogsignal(data["wave"], [])

    def _wave_to_analogsignal(self, content, dirpath):
        if "padding" in content:
            if content["padding"].size != 0:
                raise NeoReadWriteError("Cannot handle non-empty padding")
        signal = content["wData"]
        note = content["note"]
        header = content["wave_header"]
        name = str(header["bname"].decode("utf-8"))
        units = "".join([x.decode() for x in header["dataUnits"]])
        if "xUnits" in header:
            # "xUnits" is used in Versions 1, 2, 3 of .pxp files
            time_units = "".join([x.decode() for x in header["xUnits"]])
        elif "dimUnits" in header:
            # Version 5 uses "dimUnits"
            # see https://github.com/AFM-analysis/igor2/blob/43fccf51714661fb96372e8119c59e17ce01f683/igor2/binarywave.py#L501
            _time_unit_structure = header["dimUnits"].ravel()
            # For the files we've seen so far, the first element of _time_unit_structure contains the units.
            # If someone has a file for which this assumption does not hold an Exception will be raised.
            if not all([element == b"" for element in _time_unit_structure[1:]]):
                raise Exception(
                    "Neo cannot yet handle the units in this file. "
                    "Please create a new issue in the Neo issue tracker at "
                    "https://github.com/NeuralEnsemble/python-neo/issues/new/choose"
                )
            time_units = _time_unit_structure[0].decode()
        else:
            time_units = ""
        if len(time_units) == 0:
            time_units = "s"
        try:
            t_start = pq.Quantity(header["hsB"], time_units)
        except KeyError:
            t_start = pq.Quantity(header["sfB"][0], time_units)
        try:
            sampling_period = pq.Quantity(header["hsA"], time_units)
        except KeyError:
            sampling_period = pq.Quantity(header["sfA"][0], time_units)
        if self.parse_notes:
            try:
                annotations = self.parse_notes(note)
            except ValueError:
                warn("Couldn't parse notes field.")
                annotations = {"note": note}
        else:
            annotations = {"note": note}
        annotations["igor_path"] = ":".join(item.decode("utf-8") for item in dirpath)

        signal = AnalogSignal(
            signal,
            units=units,
            copy=None,
            t_start=t_start,
            sampling_period=sampling_period,
            name=name,
            file_origin=str(self.filename),
            **annotations,
        )
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
