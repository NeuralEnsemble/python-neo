from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.alphaomegarawio import AlphaOmegaRawIO


class AlphaOmegaIO(AlphaOmegaRawIO, BaseFromRaw):
    """Class for reading data from AlphaOmega MPX file"""

    def __init__(self, filename):
        AlphaOmegaRawIO.__init__(self, dirname=filename)
        BaseFromRaw.__init__(self, filename)
