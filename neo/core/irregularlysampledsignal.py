# -*- coding: utf-8 -*-
'''
This module implements :class:`IrregularlySampledSignal`, an array of analog
signals with samples taken at arbitrary time points.

:class:`IrregularlySampledSignal` inherits from :class:`basesignal.BaseSignal`
which derives from :class:`BaseNeo`, from :module:`neo.core.baseneo`, 
and from :class:`quantities.Quantity`, which in turn inherits from 
:class:`numpy.ndarray`.

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

from copy import deepcopy
import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, MergeError, merge_annotations
from neo.core.basesignal import BaseSignal
from neo.core.channelindex import ChannelIndex


def _new_IrregularlySampledSignal(cls, times, signal, units=None, time_units=None, dtype=None,
                                  copy=True, name=None, file_origin=None, description=None,
                                  annotations=None, segment=None, channel_index=None):
    '''
    A function to map IrregularlySampledSignal.__new__ to a function that
    does not do the unit checking. This is needed for pickle to work.
    '''
    iss = cls(times=times, signal=signal, units=units, time_units=time_units, 
               dtype=dtype, copy=copy, name=name, file_origin=file_origin,
               description=description, **annotations)
    iss.segment = segment
    iss.channel_index = channel_index
    return iss


class IrregularlySampledSignal(BaseSignal):
    '''
    An array of one or more analog signals with samples taken at arbitrary time points.

    A representation of one or more continuous, analog signals acquired at time
    :attr:`t_start` with a varying sampling interval. Each channel is sampled
    at the same time points.

    Inherits from :class:`quantities.Quantity`, which in turn inherits from
    :class:`numpy.ndarray`.

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
        signal = cls._rescale(signal, units=units)
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
                                               self.annotations,
                                               self.segment,
                                               self.channel_index)

    def _array_finalize_spec(self, obj):
        '''
        Set default values for attributes specific to :class:`IrregularlySampledSignal`.
        
        Common attributes are defined in
        :meth:`__array_finalize__` in :class:`basesignal.BaseSignal`),
        which is called every time a new signal is created
        and calls this method.
        '''
        self.times = getattr(obj, 'times', None)
        return obj

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_signal = cls(self.times, np.array(self), units=self.units,
                         time_units=self.times.units, dtype=self.dtype,
                         t_start=self.t_start, name=self.name,
                         file_origin=self.file_origin, description=self.description)
        new_signal.__dict__.update(self.__dict__)
        memo[id(self)] = new_signal
        for k, v in self.__dict__.items():
            try:
                setattr(new_signal, k, deepcopy(v, memo))
            except TypeError:
                setattr(new_signal, k, v)
        return new_signal

    def __repr__(self):
        '''
        Returns a string representing the :class:`IrregularlySampledSignal`.
        '''
        return '<%s(%s at times %s)>' % (self.__class__.__name__,
                                         super(IrregularlySampledSignal,
                                               self).__repr__(), self.times)

    def __getitem__(self, i):
        '''
        Get the item or slice :attr:`i`.
        '''
        obj = super(IrregularlySampledSignal, self).__getitem__(i)
        if isinstance(i, (int, np.integer)):  # a single point in time across all channels
            obj = pq.Quantity(obj.magnitude, units=obj.units)
        elif isinstance(i, tuple):
            j, k = i
            if isinstance(j, (int, np.integer)):  # a single point in time across some channels
                obj = pq.Quantity(obj.magnitude, units=obj.units)
            else:
                if isinstance(j, slice):
                    obj.times = self.times.__getitem__(j)
                elif isinstance(j, np.ndarray):
                    raise NotImplementedError("Arrays not yet supported")
                else:
                    raise TypeError("%s not supported" % type(j))
                if isinstance(k, (int, np.integer)):
                    obj = obj.reshape(-1, 1)
                    # add if channel_index
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

    def merge(self, other):
        '''
        Merge another signal into this one.

        The signal objects are concatenated horizontally
        (column-wise, :func:`np.hstack`).

        If the attributes of the two signals are not
        compatible, an Exception is raised.

        Required attributes of the signal are used.
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
        signal = self.__class__(self.times, stack, units=self.units, dtype=self.dtype,
                                copy=False, **kwargs)
        signal.segment = self.segment

        if hasattr(self, "lazy_shape"):
            signal.lazy_shape = merged_lazy_shape

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

        return signal