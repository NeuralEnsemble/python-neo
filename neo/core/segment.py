from neo.core.baseneo import BaseNeo

class Segment(BaseNeo):
    """
    A Segment is a heterogeneous container for discrete or continous data
    sharing a common clock (time basis) but not necessary the same sampling rate,
    start or end time.

    *Usage*:
    
    TODO
    
    *Required attributes/properties*:
        None
    
    *Recommended attributes/properties*:
        :name: A label for the dataset 
        :description: text description
        :file_origin: filesystem path or URL of the original data file.
        :file_datetime: the creation date and time of the original data file.
        :rec_datetime: the date and time of the original recording
        :index: TODO: WHAT IS THIS FOR?
    
    *Container of*:
        :py:class:`Epoch`
        :py:class:`EpochArray`
        :py:class:`Event`
        :py:class:`EventArray`
        :py:class:`AnalogSignal`
        :py:class:`AnalogSignalArray`
        :py:class:`IrregularlySampledSignal`
        :py:class:`Spike`
        :py:class:`SpikeTrain`

    """
    def __init__(self, file_datetime=None, rec_datetime=None, index=None,  **kargs):
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


