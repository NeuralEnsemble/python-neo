from neo.core.baseneo import BaseNeo

class RecordingChannel(BaseNeo):
    """
    A RecordingChannel is a container of AnalogSIgnal or SpikeTrain or Unit that come
    from the same logical and/or physical channel inside a Block.
    

    Usage:
    
    
    Necessary Attributes/properties:
        index (int):
    
    Recommanded Attributes/properties:
        coordinate x,y,z (quantitite array):
        name:
        description:
        file_origin:           
    
    
    Container of:
        AnalogSignal
        AnalogSignalArray
    
    Container of (many to many relationship):
        Unit
        
    """
    def __init__(self, index = 0, **kargs):
        BaseNeo.__init__(self, **kargs)
        self.index = index
        self.analogsignals = [ ]
        self.irregularysampledsignals = [ ]

