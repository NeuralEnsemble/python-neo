from neo.core.baseneo import BaseNeo

class Unit(BaseNeo):
    def __init__(self, name='', spike=[], spiketrain=[], recordingchannels=[])
        BaseNeo.__init__(self)
        self.name = name
        self.spike = spike
        self.spiketrain = spiketrain
        self.recordingchannels = recordingchannels
        return self
