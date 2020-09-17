from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.winedrrawio import WinEdrRawIO


class WinEdrIO(WinEdrRawIO, BaseFromRaw):
    """
    Class for reading data from WinEdr, a software tool written by
    John Dempster.

    WinEdr is free:
    http://spider.science.strath.ac.uk/sipbs/software.htm
    """
    _prefered_signal_group_mode = 'group-by-same-units'
    _default_group_mode_have_change_in_0_9 = True

    def __init__(self, filename):
        WinEdrRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
