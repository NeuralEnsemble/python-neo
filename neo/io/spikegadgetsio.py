from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.spikegadgetsrawio import SpikeGadgetsRawIO


class SpikeGadgetsIO(SpikeGadgetsRawIO, BaseFromRaw):
    __doc__ = SpikeGadgetsRawIO.__doc__
    def __init__(self, filename):
        SpikeGadgetsRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
