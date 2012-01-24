from neo.core.baseneo import BaseNeo

class Block(BaseNeo):
    """
    Main container gathering all the data, whether discrete or continous, for a
    given recording session.
    
    A block is not necessarily temporally homogeneous, in contrast to Segment.
    
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
        :index: integer. You can use this to define an ordering of your Block.
            It is not used by Neo in any way.
    
    *Container of*:
        :py:class:`Segment`
        :py:class:`RecordingChannelGroup`
    
    *Properties*
        list_units : descends through hierarchy and returns a list of
            :py:class:`Unit` existing in the block. This shortcut exists
            because a common analysis case is analyzing all neurons that
            you recorded in a session.
        
        list_recordingchannels: descends through hierarchy and returns
            a list of :py:class:`RecordingChannel` existing in the block.
        
        
    """
    def __init__(self, name=None, description=None, file_origin=None,
                 file_datetime=None, rec_datetime=None, index=None, 
                 **annotations):
        """Initalize a new Block."""
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        
        self.file_datetime = file_datetime
        self.rec_datetime = rec_datetime
        self.index = index
        
        self.segments = [ ]
        self.recordingchannelgroups = [ ]
        
    @property
    def list_units(self):
        """
        Return a list of all :py:class:`Unit` in a block.
        """
        units = [ ]
        for rcg in self.recordingchannelgroups:
            for rc in rcg.recordingchannel:
                for unit in rc.units:
                    if unit not in units:
                        units.append(unit)
        return units
    
    @property
    def list_recordingchannels(self):
        """
        Return a list of all :py:class:`RecordingChannel` in a block.
        """
        all_rc = [ ]
        for rcg in self.recordingchannelgroups:
            for rc in rcg.recordingchannel:
                if rc not in all_rc:
                    all_rc.append(rc)
        return all_rc

