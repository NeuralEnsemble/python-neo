from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.plexon2rawio import Plexon2RawIO


class Plexon2IO(Plexon2RawIO, BaseFromRaw):
    """
    Class for reading data from Plexon PL2 files

    The IO is based on the Plexon2RawIO, see comments for memory optimization
    in neo.rawio.plexon2rawio.Plexon2RawIO

    """

    def __init__(self, filename):
        Plexon2RawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
