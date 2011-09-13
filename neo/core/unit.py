from neo.core.baseneo import BaseNeo

class Unit(BaseNeo):
    """    
    A Unit regroups all the SpikeTrain (or Spike) objects that were emitted
    by a neuron during a Block. The spikes may come from different Segment
    within the Block, so this object is not contained in the usual 
    Block/Segment/SpikeTrain hierarchy.
    
    A Unit is linked to a list of RecordingChannel on which it was detected.
    With tetrodes, for instance, multiple channels may record the same unit.
    
    This replaces the Neuron object in the last version.
    
    Usage:
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

    
    Necessary Attributes/properties:
    
    Recommended Attributes/properties:
        name
    
    Container of:
        SpikeTrain
        Spike
    
    Container of (many to many):
       RecordingChannel
    """
    def __init__(self, **kargs):
        BaseNeo.__init__(self, **kargs)
        self.spiketrains = [ ]
        self.spikes = [ ]
        self.recordingchannels = [ ]


