from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.openephysbinaryrawio import OpenEphysBinaryRawIO


class OpenEphysBinaryIO(OpenEphysBinaryRawIO, BaseFromRaw):
    _prefered_signal_group_mode = 'group-by-same-units'
    mode = 'dir'

    def __init__(self, dirname):
        OpenEphysBinaryRawIO.__init__(self, dirname=dirname)
        BaseFromRaw.__init__(self, dirname)
