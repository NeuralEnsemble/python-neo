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
        IrregularlySampledSignal
        Spike
        SpikeTrain
        
        
        
    """
    def __init__(self, file_datetime = None,rec_datetime = None,index = None,  **kargs):
        BaseNeo.__init__(self, **kargs)
        
        self.file_datetime = file_datetime
        self.rec_datetime = rec_datetime
        self.index = index        
        
        self.epochs = [ ]
        self.epocharrays = [ ]
        self.events = [ ]
        self.eventarrays = [ ]
        self.analogsignals = [ ]
        self.analogsignalarrays = [ ]
        self.irregularlysampledsignals = [ ]
        self.spikes = [ ]
        self.spiketrains = [ ]


