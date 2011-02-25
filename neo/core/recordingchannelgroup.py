from neo.core.baseneo import BaseNeo

class RecordingChannelGroup(BaseNeo):
    def __new__(self, name='', recordchannels=[], analogsignalsarrays=[])
        self.name = name
        self.recordchannels = recordchannels
        self.analogsignalsarrays = analogsignalsarrays
        return self
