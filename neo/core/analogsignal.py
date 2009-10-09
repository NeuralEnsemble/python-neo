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
        self.t_stop  = self.t_start + len(self.signal)*self.dt
    
    def __len__(self):
        return len(self.signal)

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