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
                **kwargs):
        """Constructs new BaseAnalogSignal from data.
        
        This is called whenever a new BaseAnalogSignal is created from the
        constructor, but not when slicing.
        
        First the Quantity array is constructed from the data. Then,        
        the attributes are set from the user's arguments.
        
        __array_finalize__ is also called on the new object.
        """
        if isinstance(signal, pq.Quantity) and units:
            signal = signal.rescale(units)
        if not units and hasattr(signal, "units"):
            units = signal.units
        obj = pq.Quantity.__new__(cls, signal, units=units, dtype=dtype, copy=copy)
        obj.t_start = t_start
        obj.sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)
        return obj
    
    def __init__(self, signal, units='', dtype=None, copy=True, 
                t_start=0*pq.s, sampling_rate=None, sampling_period=None,
                **kwargs):
        """Initializes newly constructed SpikeTrain."""
        # This method is only called when constructing a new SpikeTrain,
        # not when slicing or viewing. We use the same call signature
        # as __new__ for documentation purposes. Anything not in the call
        # signature is stored in _annotations.
        
        # Calls parent __init__, which grabs universally recommended
        # attributes and sets up self._annotations        
        BaseNeo.__init__(self, **kwargs)
    
    def __array_finalize__(self, obj):
        """This is called every time a new BaseAnalogSignal is created.
        
        It is the appropriate place to set default values for attributes
        for BaseAnalogSignal constructed by slicing or viewing.
        
        User-specified values are only relevant for construction from
        constructor, and these are set in __new__. Then they are just
        copied over here.
        """        
        super(BaseAnalogSignal, self).__array_finalize__(obj)
        self.t_start = getattr(obj, 't_start', 0*pq.s)
        self.sampling_rate = getattr(obj, 'sampling_rate', None)
        
        # The additional arguments
        self._annotations = getattr(obj, '_annotations', None)
        
        # Globally recommended attributes
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)
    
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
            # update t_start
            if isinstance(i, slice):
                slice_start = i.start
            elif isinstance(i, tuple) and len(i) == 2:
                slice_start = i[0].start
            if slice_start:
                obj.t_start = self.t_start + slice_start*self.sampling_period
        return obj

    # sampling_period attribute is handled as a property on underlying rate
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
      >>> a = AnalogSignal([1,2,3])
      >>> b = AnalogSignal([4,5,6], sampling_period=42*ms)
      >>> c = AnalogSignal([1,2,3], t_start=42*ms)
      >>> d = AnalogSignal([1,2,3], t_start=42*ms, sampling_rate=0.42*kHz])
      >>> e = AnalogSignal([1,2,3], units='mV')

    Necessary Attributes/properties:
      signal : Quantity, the data itself.
      Alternatively signal can be given as an array or list, but in this
      case the keyword argument `units` must be specified.
      
      The optional `dtype` and `copy` arguments also modify the signal
      representation.
    
    Recommended Attributes/properties:
      t_start :         Quantity, time when signal begins. Default: 0.0 seconds
      
      One of the following:
      sampling_rate :   Quantity, number of samples per unit time
      sampling_period : Quantity, interval between two samples
      If neither is specified, 1.0 Hz is used.
      If both are specified, they are checked for consistency.
      (Internally this is always stored as a rate, with property access
      for period.)

      Note that the length of the signal array and the sampling rate 
      are used to calculate t_stop and duration.

    Universally recommended Attributes/properties:
      name, description, file_origin : string
    
    Any other additional arguments are assumed to be user-specific metadata
    and stored in `self._annotations`.
      >>> a = AnalogSignal([1,2,3], day='Monday')
      >>> print a._annotations['day']
      Monday
    
    Properties available on this object:
      sampling_rate, sampling_period, t_stop, duration
    
    Operations available on this object:
      == != + * /
    """
    pass
