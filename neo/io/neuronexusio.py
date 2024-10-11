from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.neuronexusrawio import NeuroNexusRawIO


class NeuroNexusIO(NeuroNexusRawIO, BaseFromRaw):
    __doc__ = NeuroNexusRawIO.__doc__
    _prefered_signal_group_mode = "group-by-same-units"

    def __init__(self, filename):
        NeuroNexusRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
