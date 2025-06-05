from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.biocamrawio import BiocamRawIO


class BiocamIO(BiocamRawIO, BaseFromRaw):
    __doc__ = BiocamRawIO.__doc__
    mode = "file"

    def __init__(self, filename, fill_gaps_strategy=None):
        BiocamRawIO.__init__(self, filename=filename, fill_gaps_strategy=fill_gaps_strategy)
        BaseFromRaw.__init__(self, filename)
