# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.micromedrawio import MicromedRawIO
from neo.core import Segment, AnalogSignal, Epoch, Event


class MicromedIO(MicromedRawIO, BaseFromRaw):
    """Class for reading/writing data from Micromed files (.trc)."""
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename):
        MicromedRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
