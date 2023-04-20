from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.mearecrawio import MEArecRawIO


class MEArecIO(MEArecRawIO, BaseFromRaw):
    __doc__ = MEArecRawIO.__doc__
    mode = 'file'

    def __init__(self, filename, load_spiketrains=True, load_recordings=True):
        MEArecRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
