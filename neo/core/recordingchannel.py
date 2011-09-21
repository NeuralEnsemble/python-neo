from neo.core.baseneo import BaseNeo

class RecordingChannel(BaseNeo):
    """
    A RecordingChannel is a container for :py:class:`AnalogSignal` objects
    that come from the same logical and/or physical channel inside a :py:class:`Block`.

    *Usage*::
    
        # Create a new RecordingChannelGroup and add to current block
        rcg = RecordingChannelGroup(channel_names=['ch1', 'ch2', 'ch3'])
        rcg.channel_indexes = [1, 2, 3]
        block.recordingchannelgroups.append(rcg)
        
        rc1 = RecordingChannel(index=1)
        rcg.recordingchannels.append(rc)
        rc2 = RecordingChannel(index=2)
        rcg.recordingchannels.append(rc)
        rc3 = RecordingChannel(index=3)
        rcg.recordingchannels.append(rc)        
        
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
    def __init__(self, index=0, coordinate=None, **kargs):
        """Initialize a new RecordingChannel."""
        # Inherited initialization
        # Sets universally recommended attributes, and places all others
        # in _annotations
        BaseNeo.__init__(self, **kargs)
        
        # Store required and recommended attributes
        self.index = index
        
        # Initialize contianers
        self.analogsignals = [ ]
        self.irregularlysampledsignals = [ ]

