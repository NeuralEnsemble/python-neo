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
    def __init__(self, file_datetime = None, rec_datetime = None, index = None,
                        **kargs):
        BaseNeo.__init__(self, **kargs)
        
        self.file_datetime.file_datetime
        self.rec_datetime = rec_datetime
        self.index = index
        
        self.segments = [ ]
        self.recordingchannelgroups = [ ]
        
    @property
    def list_units(self):
        """
        Give a list of all Unit in a block.
        """
        units = [ ]
        for rcg in self.recordingchannelgroups:
            for rc in rcg.recordingchannel:
                for unit in rc.units:
                    if unit not in units:
                        units.append(unit)
        return units


