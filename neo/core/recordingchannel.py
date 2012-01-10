from neo.core.baseneo import BaseNeo

class RecordingChannel(BaseNeo):
    """
    A RecordingChannel is a container for :py:class:`AnalogSignal` objects
    that come from the same logical and/or physical channel inside a :py:class:`Block`.
    
    Note that a RecordingChannel can belong to several :py:class:`RecordingChannelGroup`.

    *Usage* one Block with 3 Segment and 16 RecordingChannel and 48 AnalogSignal::
        
        bl = Block()
        # Create a new RecordingChannelGroup and add to current block
        rcg = RecordingChannelGroup(name = 'all channels)
        bl.recordingchannelgroups.append(rcg)
        
        for c in range(16):
            rc = RecordingChannel(index=c)
            rcg.recordingchannels.append(rc) # <- many to many relationship
            rc.recordingchannelgroups.append(rcg) # <- many to many relationship

        for s in range(3):
            seg = Segment(name = 'segment %d' %s, index = s)
            bl.segments.append(seg)
        
        for c in range(16):
            for s in range(3):
                anasig = AnalogSignal( np.rand(100000), sampling_rate = 20*pq.Hz)
                bl.segments[s].analogsignals.append(anasig)
                rcg.recordingchannels[c].analogsignals.append(anasig)
        
        
    *Required attributes/properties*:
        :index: (int) Index of the channel
    
    *Recommended attributes/properties*:
        :coordinate: (Quantity) x, y, z
        :name: string
        :description: string
        :file_origin: string
            
    *Container of*:
        :py:class:`AnalogSignal`
        :py:class:`IrregularlySampledSignal`
        
    """
    def __init__(self, index=0, coordinate=None, name=None, description=None,
                 file_origin=None, **annotations):
        """Initialize a new RecordingChannel."""
        # Inherited initialization
        # Sets universally recommended attributes, and places all others
        # in annotations
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        
        # Store required and recommended attributes
        self.index = index
        self.coordinate = coordinate
        
        # Initialize contianers
        self.analogsignals = [ ]
        self.irregularlysampledsignals = [ ]
        # Many to many relationship
        self.recordingchannelgroups = [ ]
        
        

