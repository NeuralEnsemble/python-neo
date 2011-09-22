from neo.core.baseneo import BaseNeo

class RecordingChannelGroup(BaseNeo):
    """
    A container for associated :py:class:`RecordingChannel` objects. It is
    useful for grouping channels that share something in common: for instance,
    the channels on a tetrode.

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
    """
    def __init__(self, channel_names=None, channel_indexes=None, **kargs):
        """Initialize a new RecordingChannelGroup."""
        # Inherited initialization
        # Sets universally recommended attributes, and places all others
        # in annotations
        BaseNeo.__init__(self, **kargs)

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
        self.recordingchannels = [ ]

