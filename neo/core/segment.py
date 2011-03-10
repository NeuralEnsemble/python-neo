from neo.core.baseneo import BaseNeo

class Segment(BaseNeo):
    """
    A Segment is a heterogeneous container for discrete or continous data data
    sharing a common clock (time basis) but not necessary the same sampling rate, t_start and t_stop.
    In short, a Segment is a recording may contain AnalogSignal, SpikeTrain, Event or Epoch, ...
    that share the same logical clock.

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
        Epoch
        EpochArray
        Event
        EventArray
        AnalogSignal
        AnalogSignalArray
        IrregularySampledSignal
        Spike
        SpikeTrain
        
        
        
    """
    def __init__(self, **kargs):
        BaseNeo.__init__(self, **kargs)
        self._epochs = [ ]
        self._epocharrays = [ ]
        self._events = [ ]
        self._eventarrays = [ ]
        self._analogsignals = [ ]
        self._analogsignalarrays = [ ]
        self._irregularysampledsignals = [ ]
        self._spikes = [ ]
        self._spiketrains = [ ]


