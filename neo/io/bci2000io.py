
# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.bci2000rawio import BCI2000RawIO


class BCI2000IO(BCI2000RawIO, BaseFromRaw):
    _prefered_signal_group_mode = 'split-all'

    def __init__(self, filename):
        BCI2000RawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
