from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.phyrawio import PhyRawIO


class PhyIO(PhyRawIO, BaseFromRaw):
    name = 'Phy IO'
    description = "Phy IO"
    mode = 'dir'

    def __init__(self, dirname):
        PhyRawIO.__init__(self, dirname=dirname)
        BaseFromRaw.__init__(self, dirname)
