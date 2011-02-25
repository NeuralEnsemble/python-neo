from neo.core.baseneo import BaseNeo

class Unit(BaseNeo):
    def __init__(self, name='', spike=[], spiketrain=[])
        BaseNeo.__init__(self)
        self.name = name
        self.spike = spike
        self.spiketrain = spiketrain
        return self
