# -*- coding: utf-8 -*-

from baseneo import BaseNeo

class RecordingPoint(BaseNeo):
    
    definition = """
    A :class:`RecordingPoint` is a physical location identifying the recorded data. It can
    for example be the position of the Eletrode.
    
    A :class:`RecordingPoint` regroups all the :class:`SpikeTrain` or :class:AnalogSignal objects
    within a common :class:`Block`, gathered accross several :class:`Segment`.
    
    For instance, it is useful for spikesorting when you want to detect and sort spikes 
    on many discontinued segments of signal coming from the same physical electrode 
    """
    
    __doc__ = """
    Define a particular position for recordings

    **Definition**"""+definition+"""
    

    with arguments:
    
    ``id``

    **Usage**

    **Example**

    """
    
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


