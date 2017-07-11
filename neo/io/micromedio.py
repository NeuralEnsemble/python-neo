# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.micromedrawio import MicromedRawIO
from neo.core import Segment, AnalogSignal, Epoch, Event

class MicromedIO(MicromedRawIO, BaseFromRaw):
    supported_objects = [Segment, AnalogSignal, Epoch, Event]
    readable_objects = [Segment]
    __prefered_signal_group_mode = 'group-by-same-units'
    def __init__(self, filename):
        MicromedRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
