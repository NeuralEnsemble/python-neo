# -*- coding: utf-8 -*-

import numpy



class AnalogSignal(object):
     
    definition = """An :class:`AnalogSignal` is a continuous data signal 
    acquired at time ``t_start`` at a certain sampling rate.
    """
    
    __doc__ = """
    Object to represent a continuous, analog signal

    **Definition**
    %s
    
    **Attributes**
    signal : An nd array of the data

    **Methods**    
    t() : Returns time values of each point in `signal`. This is the
        preferred way to access the time values, because it will only
        compute the values once, when it is first called. Subsequent calls
        will return a cached version.
    compute_time_vector() : Recomputes and returns time values of each point
        in `signal`.
    max() : Maximum of `signal`
    min() : Minimum of `signal`
    timeslice() : A new AnalogSignal sliced from this one.
    """ % definition
    


    def __init__(self, *arg, **karg):
        """Initialize a new AnalogSignal.
        
        Parameters
        ----------
        signal : ndarray to put in object
        channel, name, sampling_rate, t_start, t_stop
        """
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
            t_start - begining of the new AnalogSignal, in seconds.
            t_stop  - end of the new AnalogSignal, in seconds.
            
            If this signal does not contain sufficient data to meet your
            request, as much data as possible will be returned and a
            warning will be printed.
        
        Usage:
            Let's say you have a list of AnalogSignal and a list of Event,
            one Event per AnalogSignal, and you want to calculate an
            Event-triggered average. You cannot simply add together
            the AnalogSignal.signal arrays because the Event occurs at
            different times in each AnalogSignal.
            
            Instead you need to slice each AnalogSignal on the Event time
            and average together the results.
            
            slices = [ ]
            for anaSig, ev in zip(anaSigList, eventList):
                # Get a slice 100ms before and 100ms after each event
                slc = anaSig.time_slice(ev.time - .1, ev.time + .1)
                slices.append(slc.signal)
            return numpy.mean(slices, axis=0)
            
            Note that for this code to work, you will need the AnalogSignal
            to actually contain 100ms of data before and after each event.
            Otherwise an error will occur.                
        
        Implementation note:
            The closest time bin to t_start and t_stop is chosen.            
            Therefore, if the t_start you specify is not in t(), 
            then the t_start of the returned signal may differ up
            to half a sampling period.
        """       
        # Get time axis and also trigger recalculation if necessary
        t = self.t()
        
        # These kinds of checks aren't a good idea because of possible
        # floating point round-off error
        #assert t_start >= t[0], "not enough data on the beginning"
        #assert t_stop <= t[-1], "not enough data on the end"
        assert t_stop > t_start, "t_stop must be > t_start"
        
        # Find the integer indices of self.signal closest to requested limits
        # Do the checks here for 
        i_start = int(numpy.rint((t_start - self.t_start) * self.sampling_rate))
        if i_start < 0:
            print "warning: you requested data before signal starts"
            i_start = 0
        
        # Add one so that it is inclusive of t_stop
        # In the most generous case, all of the data will be included and
        # i_stop == len(self.signal), which is okay because of fancy indexing
        i_stop = int(numpy.rint((t_stop - self.t_start) * self.sampling_rate)) + 1
        if i_stop > len(self.signal):
            print "warning: you requested data after signal ended"
            i_stop = len(self.signal)
        
        # Slice the signal
        signal = self.signal[i_start:i_stop]
        
        # Create a new AnalogSignal with the specified data and the correct
        # underlying time-axis
        result = AnalogSignal(signal=signal, sampling_rate=self.sampling_rate, 
            t_start=t[i_start])
        return result