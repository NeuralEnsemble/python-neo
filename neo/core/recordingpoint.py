# -*- coding: utf-8 -*-

class RecordingPoint(object):
    
    definition = """
    A :class:`RecordingPoint` is a physical location identifying the recorded data. It can
    for example be the position of the electrode.
    
    A :class:`RecordingPoint` groups all the :class:`SpikeTrain` or :class:AnalogSignal objects
    within a common :class:`Block`, gathered accross several :class:`Segment` objects.
    
    For instance, it is useful for spike sorting when you want to detect and sort spikes 
    on many discontinuous segments of signal coming from the same physical electrode.
    """
    
    __doc__ = """
    Define a particular position for recordings

    **Definition**
    %s
    

    with arguments:
    
    ``id``

    **Usage**

    **Example**

    """ % definition
    
    def __init__(self, *arg, **karg):
        self._analogsignals = [ ]
        self._spiketrains = [ ]
        #~ self._neurons = [ ]
        self.channel = None
        
        
        if 'analogsignals' in karg :
            self._analogsignals = karg['analogsignals']
        
        if 'spiketrains' in karg :
            self._spiketrains = karg['spiketrains']
            
        if 'neurons' in karg :
            self._neurons = karg['neurons']
        
        if 'channel' in karg :
            self.channel = karg['channel']


