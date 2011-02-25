from neo.core.baseneo import BaseNeo

class Segment(BaseNeo):
    def __new__(self, name='', filedatetime=None, index=None, analogsignals=[], analogsignalsarrays=[], events=[], eventarrays=[], epoch=[], epocharrays=[]):
        self.name = name
        self.filedatetime = filedatetime
        self.index = index
        self.events = events
        self.eventarrays = eventarrays
        self.epoch = epoch
        self.epocharrays = epocharrays
        self.analogsignals = analogsignals
        self.analogsignalsarrays = analogsignalsarray
        return self
