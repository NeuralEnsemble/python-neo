from neo.core.baseneo import BaseNeo

class Unit(BaseNeo):
    """    
    A :class:`Unit` regroups all the :class:`SpikeTrain` objects that were emitted
    by a neuron during a :class:`Block`. The spikes may come from different :class:`Segment` objects
    within the :class:`Block`, so this object is not contained in the usual 
    :class:`Block`/:class:`Segment`/:class:`SpikeTrain` hierarchy.
    
    A :class:`Unit` is linked to :class:`RecordingChannelGroup` objects from which it was detected.
    With tetrodes, for instance, multiple channels may record the same unit.
    
    This replaces the :class:`Neuron` class in the previous version of Neo.
    
    *Usage*::
    
        # Store the spike times from a pyramidal neuron recorded on channel 0
        u = neo.Unit(name='pyramidal neuron')
        
        # first segment
        st1 = neo.SpikeTrain(times=[.01, 3.3, 9.3], units='sec')
        u.spiketrains.append(st1)
        
        # second segment
        st2 = neo.SpikeTrain(times=[100.01, 103.3, 109.3], units='sec')
        u.spiketrains.append(st2)
        
    
    *Required attributes/properties*:
        None
        
    *Recommended attributes/properties*:
        :name:
        :description:
        :file_origin:   
    
    *Container of*:
        :class:`SpikeTrain`
        :class:`Spike`
    
    """
    def __init__(self, name=None, description=None, file_origin=None, **annotations):
        """Initialize a new neuronal Unit (spike source)"""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        self.spiketrains = [ ]
        self.spikes = [ ]
        
        self.recordingchannelgroup = None
        




