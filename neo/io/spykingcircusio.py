from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.spykingcircusrawio import SpykingCircusRawIO


class SpykingCircusIO(SpykingCircusRawIO, BaseFromRaw):
    __doc__ = SpykingCircusRawIO.__doc__
    mode = 'dir'

    def __init__(self, dirname):
        SpykingCircusRawIO.__init__(self, dirname=dirname)
        BaseFromRaw.__init__(self, dirname)
