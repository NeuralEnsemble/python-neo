# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.winedrrawio import WinEdrRawIO


class WinEdrIO(WinEdrRawIO, BaseFromRaw):
    """
    Class for reading data from WinEdr, a software tool written by
    John Dempster.

    WinEdr is free:
    http://spider.science.strath.ac.uk/sipbs/software.htm
    """
    _prefered_signal_group_mode = 'split-all'

    def __init__(self, filename):
        WinEdrRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
