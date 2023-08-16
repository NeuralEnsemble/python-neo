from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.intanbinaryrawio import IntanBinaryRawIO


class IntanIO(IntanBinaryRawIO, BaseFromRaw):
    __doc__ = IntanBinaryRawIO.__doc__
    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, dirname):
        IntanBinaryRawIO.__init__(self, dirname=dirname)
        BaseFromRaw.__init__(self, dirname)