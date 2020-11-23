from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.oldopenephysrawio import OldOpenEphysRawIO


class OldOpenEphysIO(OldOpenEphysRawIO, BaseFromRaw):
    _prefered_signal_group_mode = 'group-by-same-units'
    mode = 'dir'

    def __init__(self, dirname):
        OpenEphysRawIO.__init__(self, dirname=dirname)
        BaseFromRaw.__init__(self, dirname)
