from neo.core.baseneo import BaseNeo

class RecordChannel(BaseNeo):
    def __new__(self, index, name, analogsignals=[])
        self.name = name
        self.analogsignals = analogsignals
        return self
