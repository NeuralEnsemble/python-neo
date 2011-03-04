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
    A representation of continuous, analog signal acquired at time ``t_start``
    at a certain sampling rate.
    
    Inherits from :class:`quantities.Quantity`, which in turn inherits from
    ``numpy.ndarray``.
    
    Usage:
      >>> from quantities import ms, kHz
      >>> a = AnalogSignal([1,2,3])
      >>> b = AnalogSignal([4,5,6], sampling_period=42*ms)
      >>> c = AnalogSignal([1,2,3], t_start=42*ms)
      >>> d = AnalogSignal([1,2,3], t_start=42*ms, sampling_rate=0.42*kHz])
      >>> e = AnalogSignal([1,2,3], units='mV')

    Attributes/properties:
      t_start :         time when signal begins
      sampling_rate :   number of samples per unit time
      sampling_period : interval between two samples (1/sampling_rate)
      duration :        signal duration (size * sampling_period)
      t_stop :          time when signal ends (t_start + duration)
      metadata :        a dictionary of user-defined attributes
    """

    def __new__(cls, signal, units='', dtype=None, copy=True, name='',
                t_start=0*pq.s, sampling_rate=None, sampling_period=None):
        """
        Create a new :class:`AnalogSignal` instance from a list or numpy array
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
        obj._annotations = {}
        return obj

    def __array_finalize__(self, obj):
        super(AnalogSignal, self).__array_finalize__(obj)
        self.t_start = getattr(obj, 't_start', 0*pq.s)
        self.sampling_rate = getattr(obj, 'sampling_rate', None)

    def __repr__(self):
        return '<AnalogSignal(%s, [%s, %s], sampling rate: %s)>' % (
             super(AnalogSignal, self).__repr__(), self.t_start, self.t_stop, self.sampling_rate)

    def __getslice__(self, i, j):
        # doesn't get called in Python 3 - __getitem__ is called instead
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

    def __eq__(self, other):
        if self.t_start != other.t_start or self.sampling_rate != other.sampling_rate:
            return False
        return super(AnalogSignal, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def _check_consistency(self, other):
        if isinstance(other, AnalogSignal):
            for attr in "t_start", "sampling_rate":
                if getattr(self, attr) != getattr(other, attr):
                    raise Exception("Inconsistent values of %s" % attr)
            # how to handle name and _annotations?

    def _copy_data_complement(self, other):
        for attr in ("t_start", "sampling_rate"): # should we copy name and annotations to the new signal?
            setattr(self, attr, getattr(other, attr))

    def _apply_operator(self, other, op):
        self._check_consistency(other)
        f = getattr(super(AnalogSignal, self), op)
        new_signal = f(other)
        new_signal._copy_data_complement(self)
        return new_signal

    def __add__(self, other):
        return self._apply_operator(other, "__add__")

    def __sub__(self, other):
        return self._apply_operator(other, "__sub__")
        
    def __mul__(self, other):
        return self._apply_operator(other, "__mul__")
        
    def __truediv__(self, other):
        return self._apply_operator(other, "__truediv__")

    __radd__ = __add__
    __rmul__ = __sub__
    
    def __rsub__(self, other):
        return self.__mul__(-1) + other

    