from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.spikeglxrawio import SpikeGLXRawIO


class SpikeGLXIO(SpikeGLXRawIO, BaseFromRaw):
    __doc__ = SpikeGLXRawIO.__doc__
    mode = 'dir'

    def __init__(self, dirname, load_sync_channel=False):
        SpikeGLXRawIO.__init__(self, dirname=dirname, load_sync_channel=load_sync_channel)
        BaseFromRaw.__init__(self, dirname)
