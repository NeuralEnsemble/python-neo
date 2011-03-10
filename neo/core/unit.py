from neo.core.baseneo import BaseNeo

class Unit(BaseNeo):
    """
    
    A Unit regroups all the SpikeTrain ( or Spike ) objects within a common Block, 
    gathered accross several Segment, that has been emitted by the same cell.
    A Unit is linked to one or several RecordingChannel (incase of tetrode for instance).
    
    Ex Neuron object in last neo version.
    
    
    
    Usage:
    
    
    Necessary Attributes/properties:
    
    Recommanded Attributes/properties:
        name

    
    Container of:
        SpikeTrain
        Spike
    
    Container of (many to many):
       RecordingChannel

    
    """
    def __init__(self, **kargs):
        BaseNeo.__init__(self, **kargs)
        self._spiketrains = [ ]
        self._spikes = [ ]
        self._recordingchannels = [ ]


