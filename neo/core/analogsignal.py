import numpy as np
import quantities as pq
from .baseneo import BaseNeo


class AnalogSignal(BaseNeo, pq.Quantity):
    """
    This class is usable only if you have installed the Quantities Python package
    BaseNeo should be the parent of every Neo class
    Quantities inherits from numpy.ndarray
    
    Usages:
    >>> a = AnalogSignal([1,2,3])
    >>> b = AnalogSignal([4,5,6], sampling_period=42.)
    >>> c = AnalogSignal([1,2,3], t_start=42.)
    >>> d = AnalogSignal([1,2,3], t_start=42., sampling_rate=1/42.])
    >>> e = AnalogSignal([1,2,3], unit='ms')

    a.signal : a numpy.ndarray view of the signal
    a.t_start : time when signal begins
    a.t_stop : t_start + len(signal) 
    a.sampling_rate : time rate between 2 values
    a.sampling_period : 1./sampling_rate

    a.metadata : a dictionary of the attributes, updated with __setattr__ and __delattr__ in BaseNeo
    """
    def __new__(subtype, signal, dtype=None, copy=True, t_start=0., sampling_rate=None, sampling_period=None, unit='ms', name=''):
        BaseNeo.__new__(subtype)
        # maybe some parameters are useless for the QuantifiedAnalogSignal use case (dtype, copy ?)
        # add recording point
        if sampling_period is None:
            if sampling_rate is None:
                sampling_rate = 1.
            
        else:
            if sampling_rate is None:
                sampling_rate = 1./sampling_period
            else:
                if sampling_period != 1./sampling_rate:
                    raise ValueError('The sampling_rate has to be 1./sampling_period')

        t_start = t_start * pq.__dict__[unit]
        sampling_rate = sampling_rate * 1./pq.__dict__[unit]

        if isinstance(signal, AnalogSignal):
            return signal

        if isinstance(signal, np.ndarray):
            new = signal.view(subtype)
            if copy: return new.copy()
            else: return new

        if isinstance(signal, str):
            # be careful about the shape: what is the dimension of my signal ?
            signal = _convert_from_string(signal)

        # now convert signal to an array
        arr = pq.Quantity(signal, unit, dtype=dtype, copy=copy)

        # added _dimensionality from quantities before the __new__ because it needs it
        subtype._dimensionality = arr._dimensionality
        subtype.t_start = t_start
        subtype.sampling_rate = float(sampling_rate)
        #subtype.sampling_period = sampling_period no redundancy, thanks

        #ret = np.ndarray.__new__(subtype, shape, arr.dtype, buffer=arr, order=order)
        ret = pq.Quantity.__new__(subtype, arr)
        ret.signal = ret.view(np.ndarray) #, dtype=arr.dtype)
        ret.t_start = t_start
        ret.sampling_rate = sampling_rate
        ret.name = name

        return ret

    def __repr__(self):
        return '<QuantifiedAnalogSignal(\n %s,\n %s, [%s, %s], sampling rate: %s)>' %(self.signal, self.unit, self.t_start, self.t_stop, self.sampling_rate)


    sampling_period = property(lambda self: 1./self.sampling_rate, lambda self, value: setattr(self, 'sampling_rate', 1./value))

    def t_stop(self):
        # do it in both senses
        return self.t_start + (len(self.signal) * self.sampling_rate)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.signal = getattr(obj, 'signal', None)
        self.t_start = getattr(obj, 't_start', None)
        self.sampling_period = getattr(obj, 'sampling_period', None)
        self.sampling_rate = getattr(obj, 'sampling_rate', None)


