# -*- coding: utf-8 -*-




class Block(object):
    
    definition = """Main container gathering all the data, whether discrete or continuous, for a given setup.
    It can be viewed as a list of :class:`Segment` objects.

    A block is not necessarily a homogeneous recording, in contrast to :class:`Segment`"""
    
    __doc__ = """
    Top level container for data.
    
    **Definition**
    %s
    
    **Usage**
    
    
    **Example**
    
    >> bl = Block( segments = [seg1 , seg2 , seg3] )
    
    **Methods**

    get_segments()
    
    
    """ % definition
    def __init__(self, *arg, **karg):
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
