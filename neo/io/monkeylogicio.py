from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.monkeylogicrawio import MonkeyLogicRawIO
import warnings

class MonkeyLogicIO(MonkeyLogicRawIO, BaseFromRaw):

    name = 'MonkeyLogicIO'

    _prefered_signal_group_mode = 'group-by-same-units'
    _prefered_units_group_mode = 'all-in-one'

    def __init__(self, filename):
        MonkeyLogicRawIO.__init__(self, filename)
        BaseFromRaw.__init__(self, filename)

    def read_block(self, block_index=0, lazy=False,
                    create_group_across_segment=None,
                    signal_group_mode=None, load_waveforms=False):

        if lazy:
            warnings.warn('Lazy loading is not supported by MonkeyLogicIO. '
                          'Ignoring `lazy=True` parameter.')

        return BaseFromRaw.read_block(self, block_index=block_index, lazy=False,
                                      create_group_across_segment=create_group_across_segment,
                                      signal_group_mode=signal_group_mode,
                                      load_waveforms=load_waveforms)
