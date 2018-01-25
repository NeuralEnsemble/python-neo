
# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.plexonrawio import PlexonRawIO


class PlexonIO(PlexonRawIO, BaseFromRaw):
    _prefered_signal_group_mode = 'split-all'

    def __init__(self, filename):
        PlexonRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
