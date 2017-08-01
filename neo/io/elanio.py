# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.elanrawio import ElanRawIO

class ElanIO(ElanRawIO, BaseFromRaw):
    __prefered_signal_group_mode = 'split-all'
    #__prefered_signal_group_mode = 'group-by-same-units'
    def __init__(self, filename):
        ElanRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
