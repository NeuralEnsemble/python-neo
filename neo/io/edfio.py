"""
IO for reading edf and edfplus files using pyedflib
https://pyedflib.readthedocs.io/en/latest/
https://github.com/holgern/pyedflib

Author: Julia Sprenger

"""

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.edfrawio import EdfRawIO


class EdfIO(EdfRawIO, BaseFromRaw):
    name = 'edf IO'
    description = "IO for reading edf and edf+ files"

    # This is an inportant choice when there are several channels.
    #   'split-all' :  1 AnalogSignal each 1 channel
    #   'group-by-same-units' : one 2D AnalogSignal for each group of channel with same units
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename=''):
        EdfRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
