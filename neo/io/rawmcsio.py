# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.rawmcsrawio import RawMCSRawIO


class RawMCSIO(RawMCSRawIO, BaseFromRaw):
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename):
        RawMCSRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
