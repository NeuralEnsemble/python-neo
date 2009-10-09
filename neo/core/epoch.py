# -*- coding: utf-8 -*-

class Epoch(object):
     
    """
    Object to represent an epoch, or discrete time events

    **Definition**
    An :class:`Epoch` is an evenement at time t, lasting for a certain duration

    with arguments:
    
    ``time`` The starting time of the epoch
    
    ``duration`` The duration of the epoch
    
    **Usage**

    **Example**

    """
    
    def __init__(self, *arg, **karg):
    
        if 'time' in karg.keys():
            self.time = karg['time']
            
        if 'duration' in karg.keys():
            self.duration = karg['duration']
            
    