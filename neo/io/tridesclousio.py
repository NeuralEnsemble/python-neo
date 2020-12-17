from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.tridesclousrawio import TridesclousRawIO


class TridesclousIO(TridesclousRawIO, BaseFromRaw):
    __doc__ = TridesclousRawIO.__doc__
    mode = 'dir'

    def __init__(self, dirname):
        TridesclousRawIO.__init__(self, dirname=dirname)
        BaseFromRaw.__init__(self, dirname)
