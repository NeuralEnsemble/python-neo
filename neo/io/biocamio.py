from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.biocamrawio import BiocamRawIO


class BiocamIO(BiocamRawIO, BaseFromRaw):
    __doc__ = BiocamRawIO.__doc__
    mode = 'file'

    def __init__(self, filename):
        BiocamRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
