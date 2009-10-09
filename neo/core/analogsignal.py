# -*- coding: utf-8 -*-

class AnalogSignal(object):
     
    """
    Object to represent an analog signal

    **Definition**
    An :class:`AnalogSignal` is a container for continuous data acquired
    at time t_start at a certain sampling rate.

    **Usage**

    **Example**

    """
    
    def __init__(self, *arg, **karg):        
        self.signal  = numpy.array(signal, float)
        self.dt      = float(dt)
        self.t_start = float(t_start)        