# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.winwcprawio import WinWcpRawIO

class WinWcpIO(WinWcpRawIO, BaseFromRaw):
    _prefered_signal_group_mode = 'split-all'
    def __init__(self, filename):
        WinWcpRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)

