from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.spikegadgetsrawio import SpikeGadgetsRawIO


class SpikeGadgetsIO(SpikeGadgetsRawIO, BaseFromRaw):
    """
    """
    def __init__(self, filename):
        SpikeGadgetsRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
