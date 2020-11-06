from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.brainvisionrawio import BrainVisionRawIO


class BrainVisionIO(BrainVisionRawIO, BaseFromRaw):
    """Class for reading data from the BrainVision product."""
    _prefered_signal_group_mode = 'group-by-same-units'
    _default_group_mode_have_change_in_0_9 = True

    def __init__(self, filename):
        BrainVisionRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
