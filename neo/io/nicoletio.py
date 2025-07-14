"""
neo.io has been split into a 2-level API:
  * neo.io: this API gives Neo objects
  * neo.rawio: this API gives raw data as they are in files.

Developers are encourage to use neo.rawio.

When this is done the neo.io can be implemented trivially
using code like shown in this file.

Author: sgarcia

"""

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.nicoletrawio import NicoletRawIO


class NicoletIO(NicoletRawIO, BaseFromRaw):
    name = "NicoleIO"
    description = "Class for reading/writing Nicolet files (.e)"

    # This is an inportant choice when there are several channels.
    #   'split-all' :  1 AnalogSignal each 1 channel
    #   'group-by-same-units' : one 2D AnalogSignal for each group of channel with same units
    _prefered_signal_group_mode = "group-by-same-units"

    def __init__(self, filename=""):
        NicoletRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
