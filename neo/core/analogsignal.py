# -*- coding: utf-8 -*-
'''
This module implements objects relating to analog signals,
:class:`BaseAnalogSignal` and its child :class:`AnalogSignal`.

:class:`AnalogSignalArray` is derived from :class:`BaseAnalogSignal` but is
defined in :module:`neo.core.analogsignalarray`.

:class:`IrregularlySampledSignal` is not derived from :class:`BaseAnalogSignal`
and is defined in :module:`neo.core.irregularlysampledsignal`.

:class:`BaseAnalogSignal` inherits from :class:`quantites.Quantity`, which
inherits from :class:`numpy.array`.
Inheritance from :class:`numpy.array` is explained here:
http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

In brief:
* Initialization of a new object from constructor happens in :meth:`__new__`.
This is where user-specified attributes are set.

* :meth:`__array_finalize__` is called for all new objects, including those
created by slicing. This is where attributes are copied over from
the old object.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo


def _get_sampling_rate(sampling_rate, sampling_period):
    '''
    Gets the sampling_rate from either the sampling_period or the
    sampling_rate, or makes sure they match if both are specified
    '''
    if sampling_period is None:
        if sampling_rate is None:
            raise ValueError("You must provide either the sampling rate or " +
                             "sampling period")
    elif sampling_rate is None:
        sampling_rate = 1.0 / sampling_period
    elif sampling_period != 1.0 / sampling_rate:
        raise ValueError('The sampling_rate has to be 1/sampling_period')
    if not hasattr(sampling_rate, 'units'):
        raise TypeError("Sampling rate/sampling period must have units")
    return sampling_rate


def _new_BaseAnalogSignal(cls, signal, units=None, dtype=None, copy=True,
                          t_start=0*pq.s, sampling_rate=None,
                          sampling_period=None, name=None, file_origin=None,
                          description=None, channel_index=None,
                          annotations=None):
    '''
    A function to map BaseAnalogSignal.__new__ to function that
        does not do the unit checking. This is needed for pickle to work.
    '''
    return cls(signal=signal, units=units, dtype=dtype, copy=copy,
               t_start=t_start, sampling_rate=sampling_rate,
               sampling_period=sampling_period, name=name,
               file_origin=file_origin, description=description,
               channel_index=channel_index,
               **annotations)


class BaseAnalogSignal(BaseNeo, pq.Quantity):
    '''
    Base class for AnalogSignal and AnalogSignalArray
    '''

    _single_parent_objects = ('Segment', 'RecordingChannel')
    _quantity_attr = 'signal'
    _necessary_attrs = (('signal', pq.Quantity, 1),
                       ('sampling_rate', pq.Quantity, 0),
                       ('t_start', pq.Quantity, 0))
    _recommended_attrs = ((('channel_index', int),) +
                          BaseNeo._recommended_attrs)

    def __new__(cls, signal, units=None, dtype=None, copy=True,
                t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                name=None, file_origin=None, description=None,
                channel_index=None, **annotations):
        '''
        Constructs new :class:`BaseAnalogSignal` from data.

        This is called whenever a new class:`BaseAnalogSignal` is created from
        the constructor, but not when slicing.

        __array_finalize__ is called on the new object.
        '''
        if units is None:
            if not hasattr(signal, "units"):
                raise ValueError("Units must be specified")
        elif isinstance(signal, pq.Quantity):
            # could improve this test, what if units is a string?
            if units != signal.units:
                signal = signal.rescale(units)

        obj = pq.Quantity(signal, units=units, dtype=dtype, copy=copy).view(cls)

        if t_start is None:
            raise ValueError('t_start cannot be None')
        obj._t_start = t_start

        obj._sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)

        obj.channel_index = channel_index
        obj.segment = None
        obj.recordingchannel = None

        return obj

    def __init__(self, signal, units=None, dtype=None, copy=True,
                 t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                 name=None, file_origin=None, description=None,
                 channel_index=None, **annotations):
        '''
        Initializes a newly constructed :class:`BaseAnalogSignal` instance.
        '''
        # This method is only called when constructing a new BaseAnalogSignal,
        # not when slicing or viewing. We use the same call signature
        # as __new__ for documentation purposes. Anything not in the call
        # signature is stored in annotations.

        # Calls parent __init__, which grabs universally recommended
        # attributes and sets up self.annotations
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

    def __reduce__(self):
        '''
        Map the __new__ function onto _new_BaseAnalogSignal, so that pickle
        works
        '''
        return _new_BaseAnalogSignal, (self.__class__,
                                       np.array(self),
                                       self.units,
                                       self.dtype,
                                       True,
                                       self.t_start,
                                       self.sampling_rate,
                                       self.sampling_period,
                                       self.name,
                                       self.file_origin,
                                       self.description,
                                       self.channel_index,
                                       self.annotations)

    def __array_finalize__(self, obj):
        '''
        This is called every time a new :class:`BaseAnalogSignal` is created.

        It is the appropriate place to set default values for attributes
        for :class:`BaseAnalogSignal` constructed by slicing or viewing.

        User-specified values are only relevant for construction from
        constructor, and these are set in __new__. Then they are just
        copied over here.
        '''
        super(BaseAnalogSignal, self).__array_finalize__(obj)
        self._t_start = getattr(obj, '_t_start', 0 * pq.s)
        self._sampling_rate = getattr(obj, '_sampling_rate', None)

        # The additional arguments
        self.annotations = getattr(obj, 'annotations', None)

        # Globally recommended attributes
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)
        self.channel_index = getattr(obj, 'channel_index', None)

    def __repr__(self):
        '''
        Returns a string representing the :class:`BaseAnalogSignal`.
        '''
        return ('<%s(%s, [%s, %s], sampling rate: %s)>' %
                (self.__class__.__name__,
                 super(BaseAnalogSignal, self).__repr__(), self.t_start,
                 self.t_stop, self.sampling_rate))

    def __getslice__(self, i, j):
        '''
        Get a slice from :attr:`i` to :attr:`j`.

        Doesn't get called in Python 3, :meth:`__getitem__` is called instead
        '''
        obj = super(BaseAnalogSignal, self).__getslice__(i, j)
        obj.t_start = self.t_start + i * self.sampling_period
        return obj

    def __getitem__(self, i):
        '''
        Get the item or slice :attr:`i`.
        '''
        obj = super(BaseAnalogSignal, self).__getitem__(i)
        if isinstance(obj, BaseAnalogSignal):
            # update t_start and sampling_rate
            slice_start = None
            slice_step = None
            if isinstance(i, slice):
                slice_start = i.start
                slice_step = i.step
            elif isinstance(i, tuple) and len(i) == 2:
                slice_start = i[0].start
                slice_step = i[0].step
            if slice_start:
                obj.t_start = self.t_start + slice_start * self.sampling_period
            if slice_step:
                obj.sampling_period *= slice_step
        return obj

    # sampling_rate attribute is handled as a property so type checking can
    # be done
    @property
    def sampling_rate(self):
        '''
        Number of samples per unit time.

        (1/:attr:`sampling_period`)
        '''
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, rate):
        '''
        Setter for :attr:`sampling_rate`
        '''
        if rate is None:
            raise ValueError('sampling_rate cannot be None')
        elif not hasattr(rate, 'units'):
            raise ValueError('sampling_rate must have units')
        self._sampling_rate = rate

    # sampling_period attribute is handled as a property on underlying rate
    @property
    def sampling_period(self):
        '''
        Interval between two samples.

        (1/:attr:`sampling_rate`)
        '''
        return 1. / self.sampling_rate

    @sampling_period.setter
    def sampling_period(self, period):
        '''
        Setter for :attr:`sampling_period`
        '''
        if period is None:
            raise ValueError('sampling_period cannot be None')
        elif not hasattr(period, 'units'):
            raise ValueError('sampling_period must have units')
        self.sampling_rate = 1. / period

    # t_start attribute is handled as a property so type checking can be done
    @property
    def t_start(self):
        '''
        Time when signal begins.
        '''
        return self._t_start

    @t_start.setter
    def t_start(self, start):
        '''
        Setter for :attr:`t_start`
        '''
        if start is None:
            raise ValueError('t_start cannot be None')
        self._t_start = start

    @property
    def duration(self):
        '''
        Signal duration

        (:attr:`size` * :attr:`sampling_period`)
        '''
        return self.shape[0] / self.sampling_rate

    @property
    def t_stop(self):
        '''
        Time when signal ends.

        (:attr:`t_start` + :attr:`duration`)
        '''
        return self.t_start + self.duration

    @property
    def times(self):
        '''
        The time points of each sample of the signal

        (:attr:`t_start` + arange(:attr:`shape`)/:attr:`sampling_rate`)
        '''
        return self.t_start + np.arange(self.shape[0]) / self.sampling_rate

    def rescale(self, units):
        '''
        Return a copy of the AnalogSignal(Array) converted to the specified
        units
        '''
        to_dims = pq.quantity.validate_dimensionality(units)
        if self.dimensionality == to_dims:
            to_u = self.units
            signal = np.array(self)
        else:
            to_u = pq.Quantity(1.0, to_dims)
            from_u = pq.Quantity(1.0, self.dimensionality)
            try:
                cf = pq.quantity.get_conversion_factor(from_u, to_u)
            except AssertionError:
                raise ValueError('Unable to convert between units of "%s" \
                                 and "%s"' % (from_u._dimensionality,
                                              to_u._dimensionality))
            signal = cf * self.magnitude
        new = self.__class__(signal=signal, units=to_u,
                             sampling_rate=self.sampling_rate)
        new._copy_data_complement(self)
        new.annotations.update(self.annotations)
        return new

    def duplicate_with_new_array(self, signal):
        '''
        Create a new :class:`BaseAnalogSignal` with the same metadata
        but different data
        '''
        #signal is the new signal
        new = self.__class__(signal=signal, units=self.units,
                             sampling_rate=self.sampling_rate)
        new._copy_data_complement(self)
        new.annotations.update(self.annotations)
        return new

    def __eq__(self, other):
        '''
        Equality test (==)
        '''
        if (self.t_start != other.t_start or
                self.sampling_rate != other.sampling_rate):
            return False
        return super(BaseAnalogSignal, self).__eq__(other)

    def __ne__(self, other):
        '''
        Non-equality test (!=)
        '''
        return not self.__eq__(other)

    def _check_consistency(self, other):
        '''
        Check if the attributes of another :class:`BaseAnalogSignal`
        are compatible with this one.
        '''
        if isinstance(other, BaseAnalogSignal):
            for attr in "t_start", "sampling_rate":
                if getattr(self, attr) != getattr(other, attr):
                    raise ValueError("Inconsistent values of %s" % attr)
            # how to handle name and annotations?

    def _copy_data_complement(self, other):
        '''
        Copy the metadata from another :class:`BaseAnalogSignal`.
        '''
        for attr in ("t_start", "sampling_rate", "name", "file_origin",
                     "description", "channel_index", "annotations"):
            setattr(self, attr, getattr(other, attr, None))

    def _apply_operator(self, other, op, *args):
        '''
        Handle copying metadata to the new :class:`BaseAnalogSignal`
        after a mathematical operation.
        '''
        self._check_consistency(other)
        f = getattr(super(BaseAnalogSignal, self), op)
        new_signal = f(other, *args)
        new_signal._copy_data_complement(self)
        return new_signal

    def __add__(self, other, *args):
        '''
        Addition (+)
        '''
        return self._apply_operator(other, "__add__", *args)

    def __sub__(self, other, *args):
        '''
        Subtraction (-)
        '''
        return self._apply_operator(other, "__sub__", *args)

    def __mul__(self, other, *args):
        '''
        Multiplication (*)
        '''
        return self._apply_operator(other, "__mul__", *args)

    def __truediv__(self, other, *args):
        '''
        Float division (/)
        '''
        return self._apply_operator(other, "__truediv__", *args)

    def __div__(self, other, *args):
        '''
        Integer division (//)
        '''
        return self._apply_operator(other, "__div__", *args)

    __radd__ = __add__
    __rmul__ = __sub__

    def __rsub__(self, other, *args):
        '''
        Backwards subtraction (other-self)
        '''
        return self.__mul__(-1, *args) + other

    def _repr_pretty_(self, pp, cycle):
        '''
        Handle pretty-printing the :class:`BaseAnalogSignal`.
        '''
        pp.text(" ".join([self.__class__.__name__,
                          "in",
                          str(self.units),
                          "with",
                          "x".join(map(str, self.shape)),
                          str(self.dtype),
                          "values",
                          ]))
        if self._has_repr_pretty_attrs_():
            pp.breakable()
            self._repr_pretty_attrs_(pp, cycle)

        def _pp(line):
            pp.breakable()
            with pp.group(indent=1):
                pp.text(line)
        if hasattr(self, "channel_index"):
            _pp("channel index: {0}".format(self.channel_index))
        for line in ["sampling rate: {0}".format(self.sampling_rate),
                     "time: {0} to {1}".format(self.t_start, self.t_stop)
                     ]:
            _pp(line)


class AnalogSignal(BaseAnalogSignal):
    '''
    A continuous analog signal.

    A representation of a continuous, analog signal acquired at time
    :attr:`t_start` at a certain sampling rate.

    Inherits from :class:`quantities.Quantity`, which in turn inherits from
    :class:`numpy.ndarray`.

    *Usage*::

        >>> from neo.core import AnalogSignal
        >>> from quantities import kHz, ms, nA, s, uV
        >>> import numpy as np
        >>>
        >>> sig0 = AnalogSignal([1, 2, 3], sampling_rate=0.42*kHz,
        ...                     units='mV')
        >>> sig1 = AnalogSignal([4, 5, 6]*nA, sampling_period=42*ms)
        >>> sig2 = AnalogSignal(np.array([1.0, 2.0, 3.0]), t_start=42*ms,
        ...                     sampling_rate=0.42*kHz, units=uV)
        >>> sig3 = AnalogSignal([1], units='V', day='Monday',
        ...                     sampling_period=1*s)
        >>>
        >>> sig3
        <AnalogSignal(array([1]) * V, [0.0 s, 1.0 s], sampling rate: 1.0 1/s)>
        >>> sig3.annotations['day']
        'Monday'
        >>> sig3[0]
        array(1) * V
        >>> sig3[::2]
        <AnalogSignal(array([1]) * V, [0.0 s, 2.0 s], sampling rate: 0.5 1/s)>

    *Required attributes/properties*:
        :signal: (quantity array 1D, numpy array 1D, or list) The data itself.
        :units: (quantity units) Required if the signal is a list or NumPy
                array, not if it is a :class:`Quantity`
        :sampling_rate: *or* :sampling_period: (quantity scalar) Number of
                                               samples per unit time or
                                               interval between two samples.
                                               If both are specified, they are
                                               checked for consistency.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :t_start: (quantity scalar) Time when signal begins.
            Default: 0.0 seconds
        :channel_index: (int) You can use this to order :class:`AnalogSignal`
            objects in an way you want.  :class:`AnalogSignalArray` and
            :class:`Unit` objects can be given indexes as well so related
            objects can be linked together.

    *Optional attributes/properties*:
        :dtype: (numpy dtype or str) Override the dtype of the signal array.
        :copy: (bool) True by default.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    *Properties available on this object*:
        :sampling_rate: (quantity scalar) Number of samples per unit time.
            (1/:attr:`sampling_period`)
        :sampling_period: (quantity scalar) Interval between two samples.
            (1/:attr:`sampling_rate`)
        :duration: (quantity scalar) Signal duration, read-only.
            (:attr:`size` * :attr:`sampling_period`)
        :t_stop: (quantity scalar) Time when signal ends, read-only.
            (:attr:`t_start` + :attr:`duration`)
        :times: (quantity 1D) The time points of each sample of the signal,
            read-only.
            (:attr:`t_start` + arange(:attr:`shape`)/:attr:`sampling_rate`)

    *Slicing*:
        :class:`AnalogSignal` objects can be sliced. When this occurs, a new
        :class:`AnalogSignal` (actually a view) is returned, with the same
        metadata, except that :attr:`sampling_period` is changed if
        the step size is greater than 1, and :attr:`t_start` is changed if
        the start index is greater than 0.  Getting a single item
        returns a :class:`~quantity.Quantity` scalar.

    *Operations available on this object*:
        == != + * /

    '''

    def __new__(cls, signal, units=None, dtype=None, copy=True,
                t_start=0*pq.s, sampling_rate=None, sampling_period=None,
                name=None, file_origin=None, description=None,
                channel_index=None, **annotations):
        '''
        Constructs new :class:`AnalogSignal` from data.

        This is called whenever a new class:`AnalogSignal` is created from
        the constructor, but not when slicing.
        '''
        obj = BaseAnalogSignal.__new__(cls, signal, units, dtype, copy,
                                       t_start, sampling_rate, sampling_period,
                                       name, file_origin, description,
                                       channel_index, **annotations)
        return obj

    def merge(self, other):
        '''
        Merging is not supported in :class:`AnalogSignal`.
        '''
        raise NotImplementedError('Cannot merge AnalogSignal objects')
