# -*- coding: utf-8 -*-


class Spike(object):
     
    """
    Object to represent a spike

    **Definition**
    An :class:`Spike` is an evenement at time t. It can be just

    with arguments:
    
    ``time`` The starting time of the epoch
    
    ``duration`` The duration of the epoch
    
    **Usage**

    **Example**

    """
    
    def __init__(self, *arg, **karg):
        
        self.time = None
        self.waveform = None
        self.freq = None
        
        if karg.has_key('time'):
            self.time = karg['time']
        
        if karg.has_key('waveform'):
            self.waveform = karg['waveform']

        if karg.has_key('freq'):
            self.freq = karg['freq']
