from __future__ import division
import numpy as np
import quantities as pq
from .baseneo import BaseNeo


def _get_sampling_rate(sampling_rate, sampling_period):
    if sampling_period is None:
        if sampling_rate is None:
            sampling_rate = 1*pq.Hz
    else:
        if sampling_rate is None:
            sampling_rate = 1./sampling_period
        else:
            if sampling_period != 1./sampling_rate:
                raise ValueError('The sampling_rate has to be 1./sampling_period')
    return sampling_rate


class AnalogSignal(BaseNeo, pq.Quantity):
    """
    This class is usable only if you have installed the Quantities Python package
    BaseNeo should be the parent of every Neo class
    Quantities inherits from numpy.ndarray
    
    Usages:
    >>> from quantities import ms
    >>> a = AnalogSignal([1,2,3])
    >>> b = AnalogSignal([4,5,6], sampling_period=42*ms)
    >>> c = AnalogSignal([1,2,3], t_start=42*ms)
    >>> d = AnalogSignal([1,2,3], t_start=42*ms, sampling_rate=1/(42*ms)])
    >>> e = AnalogSignal([1,2,3], units='mV')

    a.signal : a numpy.ndarray view of the signal
    a.t_start : time when signal begins
    a.t_stop : t_start + len(signal) 
    a.sampling_rate : time rate between 2 values
    a.sampling_period : 1./sampling_rate

    a.metadata : a dictionary of the attributes, updated with __setattr__ and __delattr__ in BaseNeo
    """

    def __new__(cls, signal, units='', dtype=None, copy=True, name='',
                t_start=0*pq.s, sampling_rate=None, sampling_period=None):
        if isinstance(signal, pq.Quantity) and units:
            signal = signal.rescale(units)
        obj = pq.Quantity.__new__(cls, signal, units=units, dtype=dtype, copy=copy)
        obj.t_start = t_start
        obj.sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)
        obj.name = name
        obj._annotations = {}
        return obj

    def __array_finalize__(self, obj):
        super(AnalogSignal, self).__array_finalize__(obj)
        self.t_start = getattr(obj, 't_start', 0*pq.s)
        self.sampling_rate = getattr(obj, 'sampling_rate', None)

    def __repr__(self):
        return '<AnalogSignal(\n %s, [%s, %s], sampling rate: %s)>' % (
            self.units, self.t_start, self.t_stop, self.sampling_rate)

    def __getslice__(self, i, j):
        # doesn't get called in Python 3 - __getitem__ called instead
        obj = super(AnalogSignal, self).__getslice__(i, j)
        obj.t_start = self.t_start + i*self.sampling_period
        return obj
    
    def __getitem__(self, i):
        obj = super(AnalogSignal, self).__getitem__(i)
        if isinstance(obj, AnalogSignal):
            obj.t_start = self.t_start + i.start*self.sampling_period
        return obj

    def _get_sampling_period(self):
        return 1./self.sampling_rate
    def _set_sampling_period(self, period):
        self.sampling_rate = 1./period
    sampling_period = property(fget=_get_sampling_period, fset=_set_sampling_period)

    @property
    def duration(self):
        return self.size/self.sampling_rate
        
    @property
    def t_stop(self):
        return self.t_start + self.duration

    @property
    def times(self):
        return self.t_start + np.arange(self.size)/self.sampling_rate
