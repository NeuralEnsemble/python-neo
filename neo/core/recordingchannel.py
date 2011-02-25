from neo.core.baseneo import BaseNeo

class RecordingChannel(BaseNeo):
    def __init__(self, index, name, analogsignals=[])
        BaseNeo.__init__(self)
        self.name = name
        self.analogsignals = analogsignals
        return self
