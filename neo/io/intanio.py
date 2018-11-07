# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.intanrawio import IntanRawIO


class IntanIO(IntanRawIO, BaseFromRaw):
    __doc__ = IntanRawIO.__doc__
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename):
        IntanRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
