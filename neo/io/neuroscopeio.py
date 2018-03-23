# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.neuroscoperawio import NeuroScopeRawIO


class NeuroScopeIO(NeuroScopeRawIO, BaseFromRaw):
    """
    Reading from Neuroscope format files.

    Ref: http://neuroscope.sourceforge.net/
    """
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename):
        NeuroScopeRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
