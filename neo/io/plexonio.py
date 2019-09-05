# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.plexonrawio import PlexonRawIO


class PlexonIO(PlexonRawIO, BaseFromRaw):
    """
    Class for reading the old data format from Plexon
    acquisition system (.plx)

    Note that Plexon now use a new format PL2 which is NOT
    supported by this IO.

    Compatible with versions 100 to 106.
    Other versions have not been tested.
    """
    _prefered_signal_group_mode = 'split-all'

    def __init__(self, filename):
        PlexonRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
