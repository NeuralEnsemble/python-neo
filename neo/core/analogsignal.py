# -*- coding: utf-8 -*-

import numpy



class AnalogSignal(object):
     
    definition = """An :class:`AnalogSignal` is a continuous data signal acquired
    at time ``t_start`` at a certain sampling rate.
    """
    
    __doc__ = """
    Object to represent a continuous, analog signal

    **Definition**
    %s

    **Usage**

    **Example**

    """ % definition
    


    def __init__(self, *arg, **karg):
        self.signal = numpy.array([], )
        if karg.has_key('signal'):
            if type(karg['signal']) == numpy.ndarray or type(karg['signal']) == numpy.memmap :
                self.signal  = karg['signal']
        

        for attr in [  'channel' , 'name', 'sampling_rate' , 't_start', 't_stop']:
            if attr in karg:
                setattr(self, attr, karg[attr])
            else:
                setattr(self, attr, None)
        
        if 'dt' in karg:
            self.sampling_rate = float(1./karg['dt'])
        
        if self.t_start is None:self.t_start = 0.
        if self.sampling_rate is None:self.sampling_rate = 1.
            
        

        if self.t_stop is None and self.sampling_rate !=0.:
            self.t_stop  = self.t_start + len(self.signal)/self.sampling_rate

        self._t = None



    #~ def __len__(self):
        #~ if self.signal is not None :
            #~ return len(self.signal)
        #~ else :
            #~ return 0
        
    def compute_time_vector(self) :
        return numpy.arange(len(self.signal), dtype = 'f8')/self.sampling_rate + self.t_start

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