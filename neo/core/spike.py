# -*- coding: utf-8 -*-

from epoch import Epoch

class Spike(Epoch):
     
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
        Epoch.__init__(self, arg, kwarg)
        self.duration = 0
    
    def waveform(self):
        pass