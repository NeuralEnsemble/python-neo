from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.bci2000rawio import BCI2000RawIO


class BCI2000IO(BCI2000RawIO, BaseFromRaw):
    """Class for reading data from a BCI2000 .dat file, either version 1.0 or 1.1"""
    _prefered_signal_group_mode = 'group-by-same-units'
    _default_group_mode_have_change_in_0_9 = True

    def __init__(self, filename):
        BCI2000RawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
