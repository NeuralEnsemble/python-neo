from neo.core.baseneo import BaseNeo

class RecordingChannelGroup(BaseNeo):
    """
    This container have sereval purpose:
      * Grouping all :py:class:`AnalogSignalArray` inside a :py:class:`Block`
        across :py:class:`Segment`
      * Grouping :py:class:`RecordingChannel` inside a :py:class:`block`. This
        case is *many to many* relation. It mean that a :py:class:`RecordingChannel`
        can belong to several group. A typical use case is tetrode (4 X :py:class:`RecordingChannel`
        inside a :py:class:`RecordingChannelGroup`).
      * Container of  :py:class:`Unit`. A neuron decharge (:py:class:`Unit`) can be seen by several
        electrodes (4 in tetrode case).
        
    *Usage 1* multi segment recording with 2 electrode array::
    
        bl = Block()
        # create a block with 3 Segment and 2 RecordingChannelGroup
        for s in range(3):
            seg = Segment(name = 'segment %d' %s, index = s)
            bl.segments.append(seg)
            
        for r in range(2):
            rcg = RecordingChannelGroup('Array probe %d' % r, channel_indexes = arange(64) )
            bl.recordingchannelgroups.append(rcg)
        
        # populating AnalogSignalArray
        for s in range(3):
            for r in range(2):
                a = AnalogSignalArray( randn(10000, 64), sampling_rate = 10*pq.kHz )
                bl.recordingchannelgroups[r].append(a)
                bl.segments[s].append(a)
        
   
    *Usage 2* grouping channel::
        
        bl = Block()
        # Create a new RecordingChannelGroup and add to current block
        rcg = RecordingChannelGroup(channel_names=['ch0', 'ch1', 'ch2'])
        rcg.channel_indexes = [0, 1, 2]
        bl.recordingchannelgroups.append(rcg)
        
        for i in range(3):
            rc = RecordingChannel(index=i)
            rcg.recordingchannels.append(rc) # <- many to many relationship
            rc.recordingchannelgroups.append(rcg) # <- many to many relationship
    
    *Usage 3* dealing with Units::
    
        bl = Block()
        rcg = RecordingChannelGroup( name = 'octotrode A')
        bl.recordingchannelgroups.append(rcg)
        
        # create several Units
        for i in range(5):
            u = Unit(name = 'unit %d' % i, description = 'after a long and hard spike sorting')
            rcg.append(u)
        
    *Required attributes*:
        None

    *Recommended attributes*:
        :channel_names: List of strings naming each channel
        :channel_indexes: List of integer indexes of each channel
        :name: string
        :description: string
        :file_origin: string
    
    *Container of*:
        :py:class:`RecordingChannel`
        :py:class:`AnalogSignalArray`
        :py:class:`Unit`
    """
    def __init__(self, channel_names=None, channel_indexes=None, name=None,
                 description=None, file_origin=None, **annotations):
        """Initialize a new RecordingChannelGroup."""
        # Inherited initialization
        # Sets universally recommended attributes, and places all others
        # in annotations
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

        # Defaults
        if channel_indexes is None:
            channel_indexes = []
        if channel_names is None:
            channel_names = []

        # Store recommended attributes
        self.channel_names = channel_names
        self.channel_indexes = channel_indexes
        
        # Initialize containers for child objects
        self.analogsignalarrays = [ ]
        self.units =[ ]
        # Many to many relationship
        self.recordingchannels = [ ]
        
        self.block = None

