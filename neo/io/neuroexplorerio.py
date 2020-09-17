from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.neuroexplorerrawio import NeuroExplorerRawIO


class NeuroExplorerIO(NeuroExplorerRawIO, BaseFromRaw):
    """Class for reading data from NeuroExplorer (.nex)"""
    _prefered_signal_group_mode = 'group-by-same-units'
    _default_group_mode_have_change_in_0_9 = True

    def __init__(self, filename):
        NeuroExplorerRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
