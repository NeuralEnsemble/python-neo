# -*- coding: utf-8 -*-
'''
This module implements :class:`AnalogSignal`, an array of analog signals.

:class:`AnalogSignal` inherits from :class:`quantites.Quantity`, which
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

import logging

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, MergeError, merge_annotations
from neo.core.channelindex import ChannelIndex

logger = logging.getLogger("Neo")


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


def _new_AnalogSignalArray(cls, signal, units=None, dtype=None, copy=True,
                          t_start=0*pq.s, sampling_rate=None,
                          sampling_period=None, name=None, file_origin=None,
                          description=None,
                          annotations=None):
    '''
    A function to map AnalogSignal.__new__ to function that
        does not do the unit checking. This is needed for pickle to work.
    '''
    return cls(signal=signal, units=units, dtype=dtype, copy=copy,
               t_start=t_start, sampling_rate=sampling_rate,
               sampling_period=sampling_period, name=name,
               file_origin=file_origin, description=description,
               **annotations)


class AnalogSignal(BaseNeo, pq.Quantity):
    '''
    Array of one or more continuous analog signals.

    A representation of several continuous, analog signals that
    have the same duration, sampling rate and start time.
    Basically, it is a 2D array: dim 0 is time, dim 1 is
    channel index

    Inherits from :class:`quantities.Quantity`, which in turn inherits from
    :class:`numpy.ndarray`.

    *Usage*::

        >>> from neo.core import AnalogSignal
        >>> import quantities as pq
        >>>
        >>> sigarr = AnalogSignal([[1, 2, 3], [4, 5, 6]], units='V',
        ...                            sampling_rate=1*pq.Hz)
        >>>
        >>> sigarr
        <AnalogSignal(array([[1, 2, 3],
              [4, 5, 6]]) * mV, [0.0 s, 2.0 s], sampling rate: 1.0 Hz)>
        >>> sigarr[:,1]
        <AnalogSignal(array([2, 5]) * V, [0.0 s, 2.0 s],
            sampling rate: 1.0 Hz)>
        >>> sigarr[1, 1]
        array(5) * V

    *Required attributes/properties*:
        :signal: (quantity array 2D, numpy array 2D, or list (data, channel))
            The data itself.
        :units: (quantity units) Required if the signal is a list or NumPy
                array, not if it is a :class:`Quantity`
        :t_start: (quantity scalar) Time when signal begins
        :sampling_rate: *or* **sampling_period** (quantity scalar) Number of
                                               samples per unit time or
                                               interval between two samples.
                                               If both are specified, they are
                                               checked for consistency.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.

    *Optional attributes/properties*:
        :dtype: (numpy dtype or str) Override the dtype of the signal array.
        :copy: (bool) True by default.

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`.

    *Properties available on this object*:
        :sampling_rate: (quantity scalar) Number of samples per unit time.
            (1/:attr:`sampling_period`)
        :sampling_period: (quantity scalar) Interval between two samples.
            (1/:attr:`quantity scalar`)
        :duration: (Quantity) Signal duration, read-only.
            (size * :attr:`sampling_period`)
        :t_stop: (quantity scalar) Time when signal ends, read-only.
            (:attr:`t_start` + :attr:`duration`)
        :times: (quantity 1D) The time points of each sample of the signal,
            read-only.
            (:attr:`t_start` + arange(:attr:`shape`[0])/:attr:`sampling_rate`)
        :channel_index:
            access to the channel_index attribute of the principal ChannelIndex
            associated with this signal.

    *Slicing*:
        :class:`AnalogSignal` objects can be sliced. When taking a single
        column (dimension 0, e.g. [0, :]) or a single element,
        a :class:`~quantities.Quantity` is returned.
        Otherwise an :class:`AnalogSignal` (actually a view) is
        returned, with the same metadata, except that :attr:`t_start`
        is changed if the start index along dimension 1 is greater than 1.

    *Operations available on this object*:
        == != + * /

    '''

    _single_parent_objects = ('Segment', 'ChannelIndex')
    _quantity_attr = 'signal'
    _necessary_attrs = (('signal', pq.Quantity, 2),
                        ('sampling_rate', pq.Quantity, 0),
                        ('t_start', pq.Quantity, 0))
    _recommended_attrs = BaseNeo._recommended_attrs

    def __new__(cls, signal, units=None, dtype=None, copy=True,
                t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                name=None, file_origin=None, description=None,
                **annotations):
        '''
        Constructs new :class:`AnalogSignal` from data.

        This is called whenever a new class:`AnalogSignal` is created from
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

        if obj.ndim == 1:
            obj.shape = (-1, 1)

        if t_start is None:
            raise ValueError('t_start cannot be None')
        obj._t_start = t_start

        obj._sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)

        obj.segment = None
        obj.channel_index = None
        return obj

    def __init__(self, signal, units=None, dtype=None, copy=True,
                 t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                 name=None, file_origin=None, description=None,
                 **annotations):
        '''
        Initializes a newly constructed :class:`AnalogSignal` instance.
        '''
        # This method is only called when constructing a new AnalogSignal,
        # not when slicing or viewing. We use the same call signature
        # as __new__ for documentation purposes. Anything not in the call
        # signature is stored in annotations.

        # Calls parent __init__, which grabs universally recommended
        # attributes and sets up self.annotations
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

    def __reduce__(self):
        '''
        Map the __new__ function onto _new_AnalogSignalArray, so that pickle
        works
        '''
        return _new_AnalogSignalArray, (self.__class__,
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
                                        self.annotations)

    def __array_finalize__(self, obj):
        '''
        This is called every time a new :class:`AnalogSignal` is created.

        It is the appropriate place to set default values for attributes
        for :class:`AnalogSignal` constructed by slicing or viewing.

        User-specified values are only relevant for construction from
        constructor, and these are set in __new__. Then they are just
        copied over here.
        '''
        super(AnalogSignal, self).__array_finalize__(obj)
        self._t_start = getattr(obj, '_t_start', 0 * pq.s)
        self._sampling_rate = getattr(obj, '_sampling_rate', None)
       
        # The additional arguments
        self.annotations = getattr(obj, 'annotations', {})

        # Globally recommended attributes
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)

        # Parents objects
        self.segment = getattr(obj, 'segment', None)
        self.channel_index = getattr(obj, 'channel_index', None)

    def __repr__(self):
        '''
        Returns a string representing the :class:`AnalogSignal`.
        '''
        return ('<%s(%s, [%s, %s], sampling rate: %s)>' %
                (self.__class__.__name__,
                 super(AnalogSignal, self).__repr__(), self.t_start,
                 self.t_stop, self.sampling_rate))

    def get_channel_index(self):
        """
        """
        if self.channel_index:
            return self.channel_index.index
        else:
            return None

    def __getslice__(self, i, j):
        '''
        Get a slice from :attr:`i` to :attr:`j`.

        Doesn't get called in Python 3, :meth:`__getitem__` is called instead
        '''
        return self.__getitem__(slice(i, j))

    def __getitem__(self, i):
        '''
        Get the item or slice :attr:`i`.
        '''
        obj = super(AnalogSignal, self).__getitem__(i)
        if isinstance(i, int):  # a single point in time across all channels
            obj = pq.Quantity(obj.magnitude, units=obj.units)
        elif isinstance(i, tuple):
            j, k = i
            if isinstance(j, int):  # extract a quantity array
                obj = pq.Quantity(obj.magnitude, units=obj.units)
            else:
                if isinstance(j, slice):
                    if j.start:
                        obj.t_start = (self.t_start +
                                       j.start * self.sampling_period)
                    if j.step:
                        obj.sampling_period *= j.step
                elif isinstance(j, np.ndarray):
                    raise NotImplementedError("Arrays not yet supported")
                    # in the general case, would need to return IrregularlySampledSignal(Array)
                else:
                    raise TypeError("%s not supported" % type(j))
                if isinstance(k, int):
                    obj = obj.reshape(-1, 1)
        elif isinstance(i, slice):
            if i.start:
                obj.t_start = self.t_start + i.start * self.sampling_period
        else:
            raise IndexError("index should be an integer, tuple or slice")
        return obj

    def __setitem__(self, i, value):
        """
        Set an item or slice defined by :attr:`i` to `value`.
        """
        # because AnalogSignals are always at least two-dimensional,
        # we need to handle the case where `i` is an integer
        if isinstance(i, int):
            i = slice(i, i + 1)
        elif isinstance(i, tuple):
            j, k = i
            if isinstance(k, int):
                i = (j, slice(k, k + 1))
        return super(AnalogSignal, self).__setitem__(i, value)

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
        Return a copy of the AnalogSignal converted to the specified
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
        Create a new :class:`AnalogSignal` with the same metadata
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
        return super(AnalogSignal, self).__eq__(other)

    def __ne__(self, other):
        '''
        Non-equality test (!=)
        '''
        return not self.__eq__(other)

    def _check_consistency(self, other):
        '''
        Check if the attributes of another :class:`AnalogSignal`
        are compatible with this one.
        '''
        if isinstance(other, AnalogSignal):
            for attr in "t_start", "sampling_rate":
                if getattr(self, attr) != getattr(other, attr):
                    raise ValueError("Inconsistent values of %s" % attr)
            # how to handle name and annotations?

    def _copy_data_complement(self, other):
        '''
        Copy the metadata from another :class:`AnalogSignal`.
        '''
        for attr in ("t_start", "sampling_rate", "name", "file_origin",
                     "description", "annotations"):
            setattr(self, attr, getattr(other, attr, None))

    def _apply_operator(self, other, op, *args):
        '''
        Handle copying metadata to the new :class:`AnalogSignal`
        after a mathematical operation.
        '''
        self._check_consistency(other)
        f = getattr(super(AnalogSignal, self), op)
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
        Handle pretty-printing the :class:`AnalogSignal`.
        '''
        pp.text("{cls} with {channels} channels of length {length}; "
                "units {units}; datatype {dtype} ".format(
                    cls=self.__class__.__name__,
                    channels=self.shape[1],
                    length=self.shape[0],
                    units=self.units.dimensionality.string,
                    dtype=self.dtype))
        if self._has_repr_pretty_attrs_():
            pp.breakable()
            self._repr_pretty_attrs_(pp, cycle)

        def _pp(line):
            pp.breakable()
            with pp.group(indent=1):
                pp.text(line)
        for line in ["sampling rate: {0}".format(self.sampling_rate),
                     "time: {0} to {1}".format(self.t_start, self.t_stop)
                     ]:
            _pp(line)


    def time_slice(self, t_start, t_stop):
        '''
        Creates a new AnalogSignal corresponding to the time slice of the
        original AnalogSignal between times t_start, t_stop. Note, that for
        numerical stability reasons if t_start, t_stop do not fall exactly on
        the time bins defined by the sampling_period they will be rounded to
        the nearest sampling bins.
        '''

        # checking start time and transforming to start index
        if t_start is None:
            i = 0
        else:
            t_start = t_start.rescale(self.sampling_period.units)
            i = (t_start - self.t_start) / self.sampling_period
            i = int(np.rint(i.magnitude))

        # checking stop time and transforming to stop index
        if t_stop is None:
            j = len(self)
        else:
            t_stop = t_stop.rescale(self.sampling_period.units)
            j = (t_stop - self.t_start) / self.sampling_period
            j = int(np.rint(j.magnitude))

        if (i < 0) or (j > len(self)):
            raise ValueError('t_start, t_stop have to be withing the analog \
                              signal duration')

        # we're going to send the list of indicies so that we get *copy* of the
        # sliced data
        obj = super(AnalogSignal, self).__getitem__(np.arange(i, j, 1))
        obj.t_start = self.t_start + i * self.sampling_period

        return obj

    def merge(self, other):
        '''
        Merge another :class:`AnalogSignal` into this one.

        The :class:`AnalogSignal` objects are concatenated horizontally
        (column-wise, :func:`np.hstack`).

        If the attributes of the two :class:`AnalogSignal` are not
        compatible, an Exception is raised.
        '''
        if self.sampling_rate != other.sampling_rate:
            raise MergeError("Cannot merge, different sampling rates")
        if self.t_start != other.t_start:
            raise MergeError("Cannot merge, different t_start")
        if self.segment != other.segment:
            raise MergeError("Cannot merge these two signals as they belong to different segments.")
        if hasattr(self, "lazy_shape"):
            if hasattr(other, "lazy_shape"):
                if self.lazy_shape[0] != other.lazy_shape[0]:
                    raise MergeError("Cannot merge signals of different length.")
                merged_lazy_shape = (self.lazy_shape[0], self.lazy_shape[1] + other.lazy_shape[1])
            else:
                raise MergeError("Cannot merge a lazy object with a real object.")
        if other.units != self.units:
            other = other.rescale(self.units)
        stack = np.hstack(map(np.array, (self, other)))
        kwargs = {}
        for name in ("name", "description", "file_origin"):
            attr_self = getattr(self, name)
            attr_other = getattr(other, name)
            if attr_self == attr_other:
                kwargs[name] = attr_self
            else:
                kwargs[name] = "merge(%s, %s)" % (attr_self, attr_other)
        merged_annotations = merge_annotations(self.annotations,
                                               other.annotations)
        kwargs.update(merged_annotations)
        signal = AnalogSignal(stack, units=self.units, dtype=self.dtype,
                              copy=False, t_start=self.t_start,
                              sampling_rate=self.sampling_rate,
                              **kwargs)
        signal.segment = self.segment
        # merge channel_index (move to ChannelIndex.merge()?)
        if self.channel_index and other.channel_index:
            signal.channel_index = ChannelIndex(
                    index=np.arange(signal.shape[1]),
                    channel_ids=np.hstack([self.channel_index.channel_ids,
                                           other.channel_index.channel_ids]),
                    channel_names=np.hstack([self.channel_index.channel_names,
                                             other.channel_index.channel_names]))
        else:
            signal.channel_index = ChannelIndex(index=np.arange(signal.shape[1]))

        if hasattr(self, "lazy_shape"):
            signal.lazy_shape = merged_lazy_shape
        return signal
