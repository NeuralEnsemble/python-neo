# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.neuroscoperawio import NeuroScopeRawIO

class NeuroScopeIO(NeuroScopeRawIO, BaseFromRaw):
    __prefered_signal_group_mode = 'group-by-same-units'
    def __init__(self, filename):
        NeuroScopeRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
