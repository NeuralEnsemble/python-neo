# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.blackrockrawio import BlackrockRawIO


class BlackrockIO(BlackrockRawIO, BaseFromRaw):
    """
    Supplementary class for reading BlackRock data using only a single nsx file.
    """
    name = 'Blackrock IO for single nsx'
    description = "This IO reads a pair of corresponding nev and nsX files of the Blackrock " \
                  "" + "(Cerebus) recording system."

    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename, **kargs):
        BlackrockRawIO.__init__(self, filename=filename, **kargs)
        BaseFromRaw.__init__(self, filename)
