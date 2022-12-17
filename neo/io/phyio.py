from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.phyrawio import PhyRawIO


class PhyIO(PhyRawIO, BaseFromRaw):
    name = 'Phy IO'
    description = "Phy IO"
    mode = 'dir'

    def __init__(self, dirname, load_amplitudes=False, load_pcs=False):
        PhyRawIO.__init__(self,
                          dirname=dirname,
                          load_amplitudes=load_amplitudes,
                          load_pcs=load_pcs)
        BaseFromRaw.__init__(self, dirname)
