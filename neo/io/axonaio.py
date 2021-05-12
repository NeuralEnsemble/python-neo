from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.axonarawio import AxonaRawIO


class AxonaIO(AxonaRawIO, BaseFromRaw):
    name = 'Axona IO'
    description = "Read raw continuous data (.bin and .set files)"

    def __init__(self, filename):
        AxonaRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
