from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.winwcprawio import WinWcpRawIO


class WinWcpIO(WinWcpRawIO, BaseFromRaw):
    """
    Class for reading data from WinWCP, a software tool written by
    John Dempster.

    WinWCP is free:
    http://spider.science.strath.ac.uk/sipbs/software.htm
    """
    _prefered_signal_group_mode = 'group-by-same-units'
    _default_group_mode_have_change_in_0_9 = True

    def __init__(self, filename):
        WinWcpRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
