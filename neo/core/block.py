# -*- coding: utf-8 -*-




class Block(object):
    
    definition = \
    """A `Block` is a container holding all the data from an experiment.
    
    It can be viewed as a list of :class: `Segment` objects.
    A block is not necessarily a homogeneous recording, in contrast
    to :class:`Segment`.
    
    A database can hold more than one Block.
    """
    
    __doc__ = """
    Top level container for data.
    
    **Definition**
    %s
    
    **Example**
    >> bl = Block( segments = [seg1 , seg2 , seg3] )
    
    **Methods**
    get_segments() : Returns list of Segment in the Block
    get_recordingpoint() : Returns list of RecordingPoint in the Block
    
    
    """ % definition
    def __init__(self, *arg, **karg):
        """Initialize a new Block.
        
        Block can be initialized with the following arguments:
        
        segments : a list of Segment to add to the block
        recordingpoints : a list of RecordingPoint add to the block
        neurons : a list of Neuron to add to the block
        """
        self._segments = [ ]
        self._recordingpoints = [ ]
        self._neurons = [ ]
        if 'segments' in karg:
            self._segments = karg['segments']
        if 'recordingpoints' in karg:
            self._recordingpoints = karg['recordingpoints']
        if 'neurons' in karg:
            self._neurons = karg['neurons']
            
    
    def get_segments(self):
        return self._segments

    def get_recordingpoints(self):
        return self._recordingpoints
