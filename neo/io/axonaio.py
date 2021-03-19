from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.axonarawio import AxonaRawIO


class AxonaIO(AxonaRawIO, BaseFromRaw):
    name = 'Axona IO'
    description = "Read raw continuous data (.bin and .set files)"

    # This is an inportant choice when there are several channels.
    #   'split-all' :  1 AnalogSignal each 1 channel
    #   'group-by-same-units' : one 2D AnalogSignal for each group of channel with same units
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename=''):
        AxonaRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)

# eof

