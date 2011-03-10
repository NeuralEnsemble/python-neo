from neo.core.baseneo import BaseNeo

class Block(BaseNeo):
    """
    Main container gathering all the data discrete or continous for a given setup.
    It can be view as a list of Segment.
    A block is not necessary a homogeneous in clock point of view recorgding contrary to Segment.
    
    Usage:
    
    
    Necessary Attributes/properties:
    
    Recommanded Attributes/properties:
        name:
        description:
        file_origin:
        file_datetime:
        rec_datetime:
        index:
    
    Container of:
        Segment
        RecordingChannelGroup
    
    """
    def __init__(self, **kargs):
        BaseNeo.__init__(self, **kargs)
        self._segments = [ ]
        self._recordingchannelgroups = [ ]
