from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.monkeylogicrawio import MonkeyLogicRawIO


class MonkeyLogicIO(MonkeyLogicRawIO, BaseFromRaw):

    name = 'MonkeyLogicIO'

    _prefered_signal_group_mode = 'group-by-same-units'
    _prefered_units_group_mode = 'all-in-one'

    def __init__(self, filename):
        MonkeyLogicRawIO.__init__(self, filename)
        BaseFromRaw.__init__(self, filename)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.header = None
        self.file.close()
