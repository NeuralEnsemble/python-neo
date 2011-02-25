from neo.core.baseneo import BaseNeo

class RecordingChannelGroup(BaseNeo):
    def __init__(self, name='', recordchannels=[], analogsignalsarrays=[])
        BaseNeo.__init__(self)
        self.name = name
        self.recordchannels = recordchannels
        self.analogsignalsarrays = analogsignalsarrays
        return self
