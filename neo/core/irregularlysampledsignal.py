# -*- coding: utf-8 -*-
'''
This module implements :class:`IrregularlySampledSignal`, an array of analog
signals with samples taken at arbitrary time points.

:class:`IrregularlySampledSignal` derives from :class:`BaseNeo`, from
:module:`neo.core.baseneo`, and from :class:`quantites.Quantity`, which
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

# needed for Python 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, MergeError, merge_annotations


def _new_IrregularlySampledSignal(cls, times, signal, units=None, time_units=None, dtype=None,
                                  copy=True, name=None, file_origin=None, description=None,
                                  annotations=None):
    '''
    A function to map IrregularlySampledSignal.__new__ to function that
    does not do the unit checking. This is needed for pickle to work.
    '''
    return cls(times=times, signal=signal, units=units, time_units=time_units, 
               dtype=dtype, copy=copy, name=name, file_origin=file_origin,
               description=description, **annotations)


class IrregularlySampledSignal(BaseNeo, pq.Quantity):
    '''
    An array of one or more analog signals with samples taken at arbitrary time points.

    A representation of one or more continuous, analog signals acquired at time
    :attr:`t_start` with a varying sampling interval. Each channel is sampled
    at the same time points.

    *Usage*::

        >>> from neo.core import IrregularlySampledSignal
        >>> from quantities import s, nA
        >>>
        >>> irsig0 = IrregularlySampledSignal([0.0, 1.23, 6.78], [1, 2, 3],
        ...                                   units='mV', time_units='ms')
        >>> irsig1 = IrregularlySampledSignal([0.01, 0.03, 0.12]*s,
        ...                                   [[4, 5], [5, 4], [6, 3]]*nA)

    *Required attributes/properties*:
        :times: (quantity array 1D, numpy array 1D, or list)
            The time of each data point. Must have the same size as :attr:`signal`.
        :signal: (quantity array 2D, numpy array 2D, or list (data, channel))
            The data itself.
        :units: (quantity units)
            Required if the signal is a list or NumPy array, not if it is
            a :class:`Quantity`.
        :time_units: (quantity units) Required if :attr:`times` is a list or
            NumPy array, not if it is a :class:`Quantity`.

    *Recommended attributes/properties*:.
        :name: (str) A label for the dataset
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.

    *Optional attributes/properties*:
        :dtype: (numpy dtype or str) Override the dtype of the signal array.
            (times are always floats).
        :copy: (bool) True by default.

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`.

    *Properties available on this object*:
        :sampling_intervals: (quantity array 1D) Interval between each adjacent
            pair of samples.
            (``times[1:] - times[:-1]``)
        :duration: (quantity scalar) Signal duration, read-only.
            (``times[-1] - times[0]``)
        :t_start: (quantity scalar) Time when signal begins, read-only.
            (``times[0]``)
        :t_stop: (quantity scalar) Time when signal ends, read-only.
            (``times[-1]``)

    *Slicing*:
        :class:`IrregularlySampledSignal` objects can be sliced. When this
        occurs, a new :class:`IrregularlySampledSignal` (actually a view) is
        returned, with the same metadata, except that :attr:`times` is also
        sliced in the same way.

    *Operations available on this object*:
        == != + * /

    '''

    _single_parent_objects = ('Segment', 'ChannelIndex')
    _quantity_attr = 'signal'
    _necessary_attrs = (('times', pq.Quantity, 1),
                        ('signal', pq.Quantity, 2))

    def __new__(cls, times, signal, units=None, time_units=None, dtype=None,
                copy=True, name=None, file_origin=None,
                description=None,
                **annotations):
        '''
        Construct a new :class:`IrregularlySampledSignal` instance.

        This is called whenever a new :class:`IrregularlySampledSignal` is
        created from the constructor, but not when slicing.
        '''
        if units is None:
            if hasattr(signal, "units"):
                units = signal.units
            else:
                raise ValueError("Units must be specified")
        elif isinstance(signal, pq.Quantity):
             # could improve this test, what if units is a string?
            if units != signal.units:
                signal = signal.rescale(units)
        if time_units is None:
            if hasattr(times, "units"):
                time_units = times.units
            else:
                raise ValueError("Time units must be specified")
        elif isinstance(times, pq.Quantity):
            # could improve this test, what if units is a string?
            if time_units != times.units:
                times = times.rescale(time_units)
        # should check time units have correct dimensions
        obj = pq.Quantity.__new__(cls, signal, units=units,
                                  dtype=dtype, copy=copy)
        if obj.ndim == 1:
            obj = obj.reshape(-1, 1)
        if len(times) != obj.shape[0]:
            raise ValueError("times array and signal array must "
                             "have same length")
        obj.times = pq.Quantity(times, units=time_units,
                                dtype=float, copy=copy)
        obj.segment = None
        obj.channel_index = None

        return obj

    def __init__(self, times, signal, units=None, time_units=None, dtype=None,
                 copy=True, name=None, file_origin=None, description=None,
                 **annotations):
        '''
        Initializes a newly constructed :class:`IrregularlySampledSignal`
        instance.
        '''
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

    def __reduce__(self):
        '''
        Map the __new__ function onto _new_IrregularlySampledSignal, so that pickle
        works
        '''
        return _new_IrregularlySampledSignal, (self.__class__,
                                               self.times, 
                                               np.array(self),
                                               self.units, 
                                               self.times.units, 
                                               self.dtype,
                                               True, 
                                               self.name, 
                                               self.file_origin,
                                               self.description,
                                               self.annotations)

    def __array_finalize__(self, obj):
        '''
        This is called every time a new :class:`IrregularlySampledSignal` is
        created.

        It is the appropriate place to set default values for attributes
        for :class:`IrregularlySampledSignal` constructed by slicing or
        viewing.

        User-specified values are only relevant for construction from
        constructor, and these are set in __new__. Then they are just
        copied over here.
        '''
        super(IrregularlySampledSignal, self).__array_finalize__(obj)
        self.times = getattr(obj, 'times', None)

        # The additional arguments
        self.annotations = getattr(obj, 'annotations', None)

        # Globally recommended attributes
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)

    def __repr__(self):
        '''
        Returns a string representing the :class:`IrregularlySampledSignal`.
        '''
        return '<%s(%s at times %s)>' % (self.__class__.__name__,
                                         super(IrregularlySampledSignal,
                                               self).__repr__(), self.times)

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
        obj = super(IrregularlySampledSignal, self).__getitem__(i)
        if isinstance(i, int):  # a single point in time across all channels
            obj = pq.Quantity(obj.magnitude, units=obj.units)
        elif isinstance(i, tuple):
            j, k = i
            if isinstance(j, int):  # a single point in time across some channels
                obj = pq.Quantity(obj.magnitude, units=obj.units)
            else:
                if isinstance(j, slice):
                    obj.times = self.times.__getitem__(j)
                elif isinstance(j, np.ndarray):
                    raise NotImplementedError("Arrays not yet supported")
                else:
                    raise TypeError("%s not supported" % type(j))
                if isinstance(k, int):
                    obj = obj.reshape(-1, 1)
        elif isinstance(i, slice):
            obj.times = self.times.__getitem__(i)
        else:
            raise IndexError("index should be an integer, tuple or slice")
        return obj


    @property
    def duration(self):
        '''
        Signal duration.

        (:attr:`times`[-1] - :attr:`times`[0])
        '''
        return self.times[-1] - self.times[0]

    @property
    def t_start(self):
        '''
        Time when signal begins.

        (:attr:`times`[0])
        '''
        return self.times[0]

    @property
    def t_stop(self):
        '''
        Time when signal ends.

        (:attr:`times`[-1])
        '''
        return self.times[-1]

    def __eq__(self, other):
        '''
        Equality test (==)
        '''
        return (super(IrregularlySampledSignal, self).__eq__(other).all() and
                (self.times == other.times).all())

    def __ne__(self, other):
        '''
        Non-equality test (!=)
        '''
        return not self.__eq__(other)

    def _apply_operator(self, other, op, *args):
        '''
        Handle copying metadata to the new :class:`IrregularlySampledSignal`
        after a mathematical operation.
        '''
        self._check_consistency(other)
        f = getattr(super(IrregularlySampledSignal, self), op)
        new_signal = f(other, *args)
        new_signal._copy_data_complement(self)
        return new_signal

    def _check_consistency(self, other):
        '''
        Check if the attributes of another :class:`IrregularlySampledSignal`
        are compatible with this one.
        '''
        # if not an array, then allow the calculation
        if not hasattr(other, 'ndim'):
            return
        # if a scalar array, then allow the calculation
        if not other.ndim:
            return
        # dimensionality should match
        if self.ndim != other.ndim:
            raise ValueError('Dimensionality does not match: %s vs %s' %
                             (self.ndim, other.ndim))
        # if if the other array does not have a times property,
        # then it should be okay to add it directly
        if not hasattr(other, 'times'):
            return

        # if there is a times property, the times need to be the same
        if not (self.times == other.times).all():
            raise ValueError('Times do not match: %s vs %s' %
                             (self.times, other.times))

    def _copy_data_complement(self, other):
        '''
        Copy the metadata from another :class:`IrregularlySampledSignal`.
        '''
        for attr in ("times", "name", "file_origin",
                     "description", "annotations"):
            setattr(self, attr, getattr(other, attr, None))

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
        return self.__mul__(-1) + other

    def _repr_pretty_(self, pp, cycle):
        '''
        Handle pretty-printing the :class:`IrregularlySampledSignal`.
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
        for line in ["sample times: {0}".format(self.times)]:
            _pp(line)

    @property
    def sampling_intervals(self):
        '''
        Interval between each adjacent pair of samples.

        (:attr:`times[1:]` - :attr:`times`[:-1])
        '''
        return self.times[1:] - self.times[:-1]

    def mean(self, interpolation=None):
        '''
        Calculates the mean, optionally using interpolation between sampling
        times.

        If :attr:`interpolation` is None, we assume that values change
        stepwise at sampling times.
        '''
        if interpolation is None:
            return (self[:-1]*self.sampling_intervals.reshape(-1, 1)).sum()/self.duration
        else:
            raise NotImplementedError

    def resample(self, at=None, interpolation=None):
        '''
        Resample the signal, returning either an :class:`AnalogSignal` object
        or another :class:`IrregularlySampledSignal` object.

        Arguments:
            :at: either a :class:`Quantity` array containing the times at
                 which samples should be created (times must be within the
                 signal duration, there is no extrapolation), a sampling rate
                 with dimensions (1/Time) or a sampling interval
                 with dimensions (Time).
            :interpolation: one of: None, 'linear'
        '''
        # further interpolation methods could be added
        raise NotImplementedError

    def rescale(self, units):
        '''
        Return a copy of the :class:`IrregularlySampledSignal` converted to the
        specified units
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
        new = self.__class__(times=self.times, signal=signal, units=to_u)
        new._copy_data_complement(self)
        new.annotations.update(self.annotations)
        return new

    def merge(self, other):
        '''
        Merge another :class:`IrregularlySampledSignal` with this one, and return the
        merged signal.

        The :class:`IrregularlySampledSignal` objects are concatenated horizontally
        (column-wise, :func:`np.hstack`).

        If the attributes of the two :class:`IrregularlySampledSignal` are not
        compatible, a :class:`MergeError` is raised.
        '''
        if not np.array_equal(self.times, other.times):
            raise MergeError("Cannot merge these two signals as the sample times differ.")
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
        signal = IrregularlySampledSignal(self.times, stack, units=self.units,
                                         dtype=self.dtype, copy=False,
                                         **kwargs)
        signal.segment = self.segment
        if hasattr(self, "lazy_shape"):
            signal.lazy_shape = merged_lazy_shape
        return signal

    def time_slice (self, t_start, t_stop):
        '''
        Creates a new :class:`IrregularlySampledSignal` corresponding to the time slice of
        the original :class:`IrregularlySampledSignal` between times
        `t_start` and `t_stop`. Either parameter can also be None
        to use infinite endpoints for the time interval.
        '''
        _t_start = t_start
        _t_stop = t_stop

        if t_start is None:
            _t_start = -np.inf
        if t_stop is None:
            _t_stop = np.inf
        indices = (self.times >= _t_start) & (self.times <= _t_stop)

        count = 0
        id_start = None
        id_stop = None
        for i in indices :
            if id_start == None :
                if i == True :
                    id_start = count
            else :
                if i == False : 
                    id_stop = count
                    break
            count += 1
        
        new_st = self[id_start:id_stop]

        return new_st