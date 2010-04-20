# -*- coding: utf-8 -*-


class Spike(object):
     
    """
    Object to represent a spike

    **Definition**
    An :class:`Spike` is an evenement at time t with a waveform.

    with arguments:
    
    ``time`` The spike time
    
    ``waveform`` An 1D array of the waveform
    
    ``sampling_rate`` The waveform sampling rate
    
    **Usage**

    **Example**

    """
    
    def __init__(self, *arg, **karg):
        
        self.time = None
        self.waveform = None
        self.sampling_rate = None
        
        if karg.has_key('time'):
            self.time = karg['time']
        
        if karg.has_key('waveform'):
            self.waveform = karg['waveform']

        if karg.has_key('sampling_rate'):
            self.sampling_rate = karg['sampling_rate']
