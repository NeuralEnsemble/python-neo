from neo.core.baseneo import BaseNeo

class Unit(BaseNeo):
    """    
    A :class:`Unit` regroups all the :class:`SpikeTrain` objects that were emitted
    by a neuron during a :class:`Block`. The spikes may come from different :class:`Segment` objects
    within the :class:`Block`, so this object is not contained in the usual 
    :class:`Block`/:class:`Segment`/:class:`SpikeTrain` hierarchy.
    
    A :class:`Unit` is linked to a list of :class:`RecordingChannel` objects from which it was detected.
    With tetrodes, for instance, multiple channels may record the same unit.
    
    This replaces the :class:`Neuron` class in the previous version of Neo.
    
    *Usage*::
    
        # Store the spike times from a pyramidal neuron recorded on channel 0
        u = neo.Unit(name='pyramidal neuron')
        
        # first segment
        st1 = neo.SpikeTrain(times=[.01, 3.3, 9.3], units='sec')
        u._spiketrains.append(st1)
        
        # second segment
        st2 = neo.SpikeTrain(times=[100.01, 103.3, 109.3], units='sec')
        u._spiketrains.append(st2)
        
        # channel info
        rc = RecordginChannel(index = 0)
        u._recordingchannels.append(rc)

    
    *Required attributes/properties*:
        None
        
    *Recommended attributes/properties*:
        :name:
        :description:
        :file_origin:   
    
    *Container of*:
        :class:`SpikeTrain`
        :class:`Spike`
    
    *Container of (many to many)*:
       :class:`RecordingChannel`

    """
    def __init__(self, **kargs):
        BaseNeo.__init__(self, **kargs)
        self.spiketrains = [ ]
        self.spikes = [ ]
        self.recordingchannels = [ ]




