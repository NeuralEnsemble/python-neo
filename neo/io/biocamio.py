from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.biocamrawio import BiocamRawIO


class BiocamIO(BiocamRawIO, BaseFromRaw):
    __doc__ = BiocamRawIO.__doc__
    mode = "file"

    def __init__(self, filename, true_zeroes=False, use_synthetic_noise=False):
        BiocamRawIO.__init__(self, filename=filename, true_zeroes=true_zeroes,
                             use_synthetic_noise=use_synthetic_noise)
        BaseFromRaw.__init__(self, filename)
