from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.maxwellrawio import MaxwellRawIO


class MaxwellIO(MaxwellRawIO, BaseFromRaw):
    mode = 'file'

    def __init__(self, filename):
        MaxwellRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
