# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.spike2rawio import Spike2RawIO


class Spike2IO(Spike2RawIO, BaseFromRaw):
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename):
        Spike2RawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
