# -*- coding: utf-8 -*-

import numpy

class AnalogSignal(object):
     
    """
    Object to represent an analog signal

    **Definition**
    An :class:`AnalogSignal` is a container for continuous data acquired
    at time t_start at a certain sampling rate.

    **Usage**

    **Example**

    """
    
    label   = None
    t_start = 0
    
    def __init__(self, *arg, **karg):
        if karg.has_key('signal'):
            if type(karg['signal']) == numpy.ndarray or type(karg['signal']) == numpy.memmap :
                self.signal  = karg['signal']
            else : 
                numpy.array(karg['signal'], dtype='float32')
        if karg.has_key('dt'):
            self.freq = float(1./karg['dt'])
        if karg.has_key('freq'):
            self.freq = karg['freq']
        
        if karg.has_key('t_start'):
            self.t_start = float(karg['t_start'])
            self.t_stop  = self.t_start + len(self.signal)/self.freq
        else :
            self.t_start = 0.
            self.t_stop = 0.
        
    def __len__(self):
        return len(self.signal)
        
    def t(self) :
        return numpy.arange(len(self.signal), dtype = 'f')/self.freq + self.t_start

    def max(self):
        return self.signal.max()

    def min(self):
        return self.signal.min()
    
    def mean(self):
        return numpy.mean(self.signal)
    
    def time_slice(self, t_start, t_stop):
        """ 
        Return a new AnalogSignal obtained by slicing between t_start and t_stop
        
        Inputs:
            t_start - begining of the new SpikeTrain, in ms.
            t_stop  - end of the new SpikeTrain, in ms.
        
        See also:
            interval_slice
        """
        assert t_start >= self.t_start
        assert t_stop <= self.t_stop
        assert t_stop > t_start
        
        t = self.time_axis()
        i_start = int((t_start-self.t_start)/self.dt)
        i_stop = int((t_stop-self.t_start)/self.dt)
        signal = self.signal[i_start:i_stop]
        result = AnalogSignal(signal, self.dt, t_start, t_stop)
        return result