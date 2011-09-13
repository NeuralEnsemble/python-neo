from __future__ import division
import numpy as np
import quantities as pq
from .analogsignal import BaseAnalogSignal, AnalogSignal, _get_sampling_rate


class AnalogSignalArray(BaseAnalogSignal):
    """
    A representation of several continuous, analog signals that
    have the same duration, sampling rate and t_start.
    Basically, it is a 2D array like AnalogSignal.
      Dim 0 is time
      Dim 1 is channel index
    
    Inherits from :class:`quantities.Quantity`, which in turn inherits from
    ``numpy.ndarray``.
    
    Usage:

    Necessary Attributes/properties:
        t_start :         time when signal begins
        sampling_rate :   number of samples per unit time
      
      
    

    Properties:
        sampling_period : interval between two samples (1/sampling_rate)
        duration :        signal duration (size * sampling_period)
        t_stop :          time when signal ends (t_start + duration)
        
    Recommanded Attributes/properties:
        name
        description
        file_origin

    """

    def __new__(cls, signal, units='', dtype=None, copy=True, t_start=0*pq.s,
                sampling_rate=None, sampling_period=None, 
                name=None, file_origin = None, description = None,
                ):
        """
        Create a new :class:`AnalogSignalArray` instance from a list or numpy array
        of numerical values, or from a Quantity array.
        """
        if isinstance(signal, pq.Quantity) and units:
            signal = signal.rescale(units)
        if not units and hasattr(signal, "units"):
            units = signal.units
        obj = pq.Quantity.__new__(cls, signal, units=units, dtype=dtype, copy=copy)
        obj.t_start = t_start
        obj.sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)
        obj.name = name
        obj.file_origin = file_origin
        obj.description = description
        obj._annotations = {}
        return obj

    def __getslice__(self, i, j):
        raise Exception("%d %d" % (i,j))
        # doesn't get called in Python 3 - __getitem__ is called instead
#        obj = super(BaseAnalogSignal, self).__getslice__(i, j)
#        obj.t_start = self.t_start + i*self.sampling_period
#        return obj

    def __getitem__(self, i):
        obj = super(BaseAnalogSignal, self).__getitem__(i)
        if isinstance(i, int):
            return obj
        elif isinstance(i, tuple):
            j,k = i
            if isinstance(k, int): # extract an AnalogSignal
                if isinstance(j, slice):
                    obj = AnalogSignal(obj, sampling_rate=self.sampling_rate)
                    if j.start:
                        obj.t_start = self.t_start + j.start*self.sampling_period
                return obj
            elif isinstance(j, int): # extract a quantity array
                obj = pq.Quantity(np.array(obj), units=obj.units) # should be a better way to do this
                return obj
            else:
                return obj
        else:
            raise IndexError("index should be an integer or a tuple")
