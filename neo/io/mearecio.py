from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.mearecrawio import MEArecRawIO


class MEArecIO(MEArecRawIO, BaseFromRaw):
    __doc__ = MEArecRawIO.__doc__
    mode = "file"

    def __init__(self, filename, load_spiketrains=True, load_analogsignal=True):
        MEArecRawIO.__init__(
            self, filename=filename, load_spiketrains=load_spiketrains, load_analogsignal=load_analogsignal
        )
        BaseFromRaw.__init__(self, filename)
