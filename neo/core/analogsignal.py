"""This module defines objects relating to analog signals.

For documentation on these objects, which are imported into the base
neo namespace, see:
    neo.AnalogSignal
    neo.AnalogSignalArray
"""

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


class BaseAnalogSignal(BaseNeo, pq.Quantity):
    """
    Base class for AnalogSignal and AnalogSignalArray
    """

    def __new__(cls, signal, units='', dtype=None, copy=True, 
                t_start=0*pq.s, sampling_rate=None, sampling_period=None,
                name=None, file_origin = None, description = None,
                ):
        """
        Create a new :class:`BaseAnalogSignal` instance from a list or numpy array
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

    def __array_finalize__(self, obj):
        super(BaseAnalogSignal, self).__array_finalize__(obj)
        self.t_start = getattr(obj, 't_start', 0*pq.s)
        self.sampling_rate = getattr(obj, 'sampling_rate', None)

    def __repr__(self):
        return '<BaseAnalogSignal(%s, [%s, %s], sampling rate: %s)>' % (
             super(BaseAnalogSignal, self).__repr__(), self.t_start, self.t_stop, self.sampling_rate)

    def __getslice__(self, i, j):
        # doesn't get called in Python 3 - __getitem__ is called instead
        obj = super(BaseAnalogSignal, self).__getslice__(i, j)
        obj.t_start = self.t_start + i*self.sampling_period
        return obj
    
    def __getitem__(self, i):
        obj = super(BaseAnalogSignal, self).__getitem__(i)
        if isinstance(obj, BaseAnalogSignal):
            obj.t_start = self.t_start + i.start*self.sampling_period
        return obj

    def _get_sampling_period(self):
        return 1./self.sampling_rate
    def _set_sampling_period(self, period):
        self.sampling_rate = 1./period
    sampling_period = property(fget=_get_sampling_period, fset=_set_sampling_period)

    @property
    def duration(self):
        return self.shape[0]/self.sampling_rate
        
    @property
    def t_stop(self):
        return self.t_start + self.duration

    @property
    def times(self):
        return self.t_start + np.arange(self.shape[0])/self.sampling_rate
    
    def copy_except_signal(self, signal):
        #signal is the new signal
        new = AnalogSignal(signal = signal, units= self.units)
        new._copy_data_complement(self)
        new._annotations.update(self._annotations)
        return new

    def __eq__(self, other):
        if self.t_start != other.t_start or self.sampling_rate != other.sampling_rate:
            return False
        return super(BaseAnalogSignal, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def _check_consistency(self, other):
        if isinstance(other, BaseAnalogSignal):
            for attr in "t_start", "sampling_rate":
                if getattr(self, attr) != getattr(other, attr):
                    raise Exception("Inconsistent values of %s" % attr)
            # how to handle name and _annotations?

    def _copy_data_complement(self, other):
        for attr in ("t_start", "sampling_rate"): # should we copy name and annotations to the new signal?
            setattr(self, attr, getattr(other, attr))

    def _apply_operator(self, other, op):
        self._check_consistency(other)
        f = getattr(super(BaseAnalogSignal, self), op)
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

    
class AnalogSignal(BaseAnalogSignal):
    """
    A representation of continuous, analog signal acquired at time ``t_start``
    at a certain sampling rate.
    
    Inherits from :class:`quantities.Quantity`, which in turn inherits from
    ``numpy.ndarray``.
    
    Usage:
      >>> from quantities import ms, kHz
      >>> a = BaseAnalogSignal([1,2,3])
      >>> b = BaseAnalogSignal([4,5,6], sampling_period=42*ms)
      >>> c = BaseAnalogSignal([1,2,3], t_start=42*ms)
      >>> d = BaseAnalogSignal([1,2,3], t_start=42*ms, sampling_rate=0.42*kHz])
      >>> e = BaseAnalogSignal([1,2,3], units='mV')

    Necessary Attributes/properties:
      t_start :         time when signal begins
      sampling_rate :   number of samples per unit time
      sampling_period : interval between two samples (1/sampling_rate)
      duration :        signal duration (size * sampling_period)
      t_stop :          time when signal ends (t_start + duration)
      
    Recommanded Attributes/properties:
      name
      description
      file_origin
    """
    pass
