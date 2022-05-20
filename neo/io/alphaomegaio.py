from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.alphaomegarawio import AlphaOmegaRawIO


class AlphaOmegaIO(AlphaOmegaRawIO, BaseFromRaw):
    """Class for reading data from AlphaOmega MPX file"""

    __doc__ = AlphaOmegaRawIO.__doc__

    def __init__(self, filename, lsx_files=None, prune_channels=True):
        AlphaOmegaRawIO.__init__(
            self,
            dirname=filename,
            lsx_files=lsx_files,
            prune_channels=prune_channels,
        )
        BaseFromRaw.__init__(self, filename)
