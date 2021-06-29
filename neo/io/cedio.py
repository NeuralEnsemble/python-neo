from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.cedrawio import CedRawIO


class CedIO(CedRawIO, BaseFromRaw):
    __doc__ = CedRawIO.__doc__

    def __init__(self, filename, entfile=None, posfile=None):
        CedRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
