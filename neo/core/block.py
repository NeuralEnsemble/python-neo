# -*- coding: utf-8 -*-




class Block(object):
    
    definition = """Main container gathering all the data discrete or continous for a given setup.
    It can be view as a list of :class:`Segment`.

    A block is not necessary a homogeneous recorging contrary to :class:`Segment`"""
    
    __doc__ = """
    Top level container for data.
    
    **Definition**"""+ definition+"""
    **Usage**
    
    
    **Example**
    
    >> bl = Block( segments = [seg1 , seg2 , seg3] )
    
    **Methods**

    get_segments()
    
    
    """
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
