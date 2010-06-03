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
    


    def __init__(self, *arg, **karg):
        self.label   = None
        self.t_start = 0.
        self.sampling_rate = 1.
        self.signal = numpy.array([], dtype='float32')
        self._t = None
        self.channel = None
    
        if karg.has_key('signal'):
            if type(karg['signal']) == numpy.ndarray or type(karg['signal']) == numpy.memmap :
                self.signal  = karg['signal']
        if karg.has_key('dt'):
            self.sampling_rate = float(1./karg['dt'])
        if karg.has_key('sampling_rate'):
            self.sampling_rate = karg['sampling_rate']
        if karg.has_key('t_start'):
            self.t_start = float(karg['t_start'])
        self.t_stop  = self.t_start + len(self.signal)/self.sampling_rate

        if 'channel' in karg :
            self.channel = karg['hannel']

    #~ def __len__(self):
        #~ if self.signal is not None :
            #~ return len(self.signal)
        #~ else :
            #~ return 0
        
    def compute_time_vector(self) :
        return numpy.arange(len(self.signal), dtype = 'f')/self.sampling_rate + self.t_start

    def t(self):
        if self._t==None:
            self._t=self.compute_time_vector()
        return self._t

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
            t_start - begining of the new AnalogSignal, in ms.
            t_stop  - end of the new AnalogSignal, in ms.
        
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