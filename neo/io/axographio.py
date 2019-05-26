# -*- coding: utf-8 -*-
"""
..............
"""

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.axographrawio import AxographRawIO


class AxographIO(AxographRawIO, BaseFromRaw):
    name = 'AxographIO'
    description = 'This IO reads .axgd/.axgx files created with AxoGraph'

    _prefered_signal_group_mode = 'split-all'

    def __init__(self, filename='', force_single_segment=False):
        AxographRawIO.__init__(self, filename=filename, force_single_segment=force_single_segment)
        BaseFromRaw.__init__(self, filename)
