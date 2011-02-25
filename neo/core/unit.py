from neo.core.baseneo import BaseNeo

class Unit(BaseNeo):
    def __new__(self, name='', spike=[], spiketrain=[])
        self.name = name
        self.spike = spike
        self.spiketrain = spiketrain
        return self
