# -*- coding: utf-8 -*-
from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.tdtrawio import TdtRawIO

class TdtIO(TdtRawIO, BaseFromRaw):
    _prefered_signal_group_mode = 'split-all'
    mode = 'dir'
    def __init__(self, dirname):
        TdtRawIO.__init__(self, dirname=dirname)
        BaseFromRaw.__init__(self, dirname)
