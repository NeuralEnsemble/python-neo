from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.neuronexusrawio import NeuronexusRawIO


class NeuronexusIO(NeuronexusRawIO, BaseFromRaw):
    __doc__ = NeuronexusRawIO.__doc__
    _prefered_signal_group_mode = "group-by-same-units"

    def __init__(self, filename):
        NeuronexusRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)