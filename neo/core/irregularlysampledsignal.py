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

from copy import deepcopy, copy

try:
    import scipy.signal
except ImportError as err:
    HAVE_SCIPY = False
else:
    HAVE_SCIPY = True

import numpy as np
import quantities as pq

from neo.core.baseneo import MergeError, merge_annotations, intersect_annotations
from neo.core.basesignal import BaseSignal
from neo.core.analogsignal import AnalogSignal
from neo.core.dataobject import DataObject


def _new_IrregularlySampledSignal(cls, times, signal, units=None, time_units=None, dtype=None,
                                  copy=True, name=None, file_origin=None, description=None,
                                  array_annotations=None, annotations=None, segment=None):
    '''
    A function to map IrregularlySampledSignal.__new__ to a function that
    does not do the unit checking. This is needed for pickle to work.
    '''
    iss = cls(times=times, signal=signal, units=units, time_units=time_units, dtype=dtype,
              copy=copy, name=name, file_origin=file_origin, description=description,
              array_annotations=array_annotations, **annotations)
    iss.segment = segment
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
        :array_annotations: (dict) Dict mapping strings to numpy arrays containing annotations \
                                   for all data points

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

    _parent_objects = ('Segment',)
    _parent_attrs = ('segment',)
    _quantity_attr = 'signal'
    _necessary_attrs = (('times', pq.Quantity, 1), ('signal', pq.Quantity, 2))

    def __new__(cls, times, signal, units=None, time_units=None, dtype=None, copy=True, name=None,
                file_origin=None, description=None, array_annotations=None, **annotations):
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
        obj = pq.Quantity.__new__(cls, signal, units=units, dtype=dtype, copy=copy)
        if obj.ndim == 1:
            obj = obj.reshape(-1, 1)
        if len(times) != obj.shape[0]:
            raise ValueError("times array and signal array must "
                             "have same length")
        obj.times = pq.Quantity(times, units=time_units, dtype=float, copy=copy)
        obj.segment = None

        return obj

    def __init__(self, times, signal, units=None, time_units=None, dtype=None, copy=True,
                 name=None, file_origin=None, description=None, array_annotations=None,
                 **annotations):
        '''
        Initializes a newly constructed :class:`IrregularlySampledSignal`
        instance.
        '''
        DataObject.__init__(self, name=name, file_origin=file_origin, description=description,
                            array_annotations=array_annotations, **annotations)

    def __reduce__(self):
        '''
        Map the __new__ function onto _new_IrregularlySampledSignal, so that pickle
        works
        '''
        return _new_IrregularlySampledSignal, (self.__class__, self.times, np.array(self),
                                               self.units, self.times.units, self.dtype, True,
                                               self.name, self.file_origin, self.description,
                                               self.array_annotations, self.annotations,
                                               self.segment)

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

    def __repr__(self):
        '''
        Returns a string representing the :class:`IrregularlySampledSignal`.
        '''
        return '<{}({} at times {})>'.format(
            self.__class__.__name__, super().__repr__(), self.times)

    def __getitem__(self, i):
        '''
        Get the item or slice :attr:`i`.
        '''
        if isinstance(i, (int, np.integer)):  # a single point in time across all channels
            obj = super().__getitem__(i)
            obj = pq.Quantity(obj.magnitude, units=obj.units)
        elif isinstance(i, tuple):
            obj = super().__getitem__(i)
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
                obj.array_annotations = deepcopy(self.array_annotations_at_index(k))
        elif isinstance(i, slice):
            obj = super().__getitem__(i)
            obj.times = self.times.__getitem__(i)
            obj.array_annotations = deepcopy(self.array_annotations)
        elif isinstance(i, np.ndarray):
            # Indexing of an IrregularlySampledSignal is only consistent if the resulting
            # number of samples is the same for each trace. The time axis for these samples is not
            # guaranteed to be continuous, so returning a Quantity instead of an
            # IrregularlySampledSignal here.
            new_time_dims = np.sum(i, axis=0)
            if len(new_time_dims) and all(new_time_dims == new_time_dims[0]):
                obj = np.asarray(self).T.__getitem__(i.T)
                obj = obj.T.reshape(self.shape[1], -1).T
                obj = pq.Quantity(obj, units=self.units)
            else:
                raise IndexError("indexing of an IrregularlySampledSignal needs to keep the same "
                                 "number of sample for each trace contained")
        else:
            raise IndexError("index should be an integer, tuple, slice or boolean numpy array")
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
        if (isinstance(other, IrregularlySampledSignal) and not (self.times == other.times).all()):
            return False
        return super().__eq__(other)

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
            raise ValueError('Dimensionality does not match: {} vs {}'.format(
                self.ndim, other.ndim))
        # if if the other array does not have a times property,
        # then it should be okay to add it directly
        if not hasattr(other, 'times'):
            return

        # if there is a times property, the times need to be the same
        if not (self.times == other.times).all():
            raise ValueError('Times do not match: {} vs {}'.format(self.times, other.times))

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
                "units {units}; datatype {dtype} ".format(cls=self.__class__.__name__,
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

        for line in ["sample times: {}".format(self.times)]:
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
            return (self[:-1] * self.sampling_intervals.reshape(-1, 1)).sum() / self.duration
        else:
            raise NotImplementedError

    def resample(self, sample_count, **kwargs):
        """
        Resample the data points of the signal.
        This method interpolates the signal and returns a new signal with a fixed number of
        samples defined by `sample_count`.
        This function is a wrapper of scipy.signal.resample and accepts the same set of keyword
        arguments, except for specifying the axis of resampling which is fixed to the first axis
        here, and the sample positions. .

        Parameters
        ----------
        sample_count: int
            Number of desired samples. The resulting signal starts at the same sample as the
            original and is sampled regularly.

        Returns
        -------
        resampled_signal: :class:`AnalogSignal`
            New instance of a :class:`AnalogSignal` object containing the resampled data points.
            The original :class:`AnalogSignal` is not modified.
        """

        if not HAVE_SCIPY:
            raise ImportError('Resampling requires availability of scipy.signal')

        # Resampling is only permitted along the time axis (axis=0)
        if 'axis' in kwargs:
            kwargs.pop('axis')
        if 't' in kwargs:
            kwargs.pop('t')

        resampled_data, resampled_times = scipy.signal.resample(self.magnitude, sample_count,
                                                                t=self.times.magnitude,
                                                                axis=0, **kwargs)

        new_sampling_rate = (sample_count - 1) / self.duration
        resampled_signal = AnalogSignal(resampled_data, units=self.units, dtype=self.dtype,
                                        t_start=self.t_start,
                                        sampling_rate=new_sampling_rate,
                                        array_annotations=self.array_annotations.copy(),
                                        **self.annotations.copy())

        # since the number of channels stays the same, we can also copy array annotations here
        resampled_signal.array_annotations = self.array_annotations.copy()
        return resampled_signal

    def time_slice(self, t_start, t_stop):
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
        for i in indices:
            if id_start is None:
                if i:
                    id_start = count
            else:
                if not i:
                    id_stop = count
                    break
            count += 1

        # Time slicing should create a deep copy of the object
        new_st = deepcopy(self[id_start:id_stop])

        return new_st

    def time_shift(self, t_shift):
        """
        Shifts a :class:`IrregularlySampledSignal` to start at a new time.

        Parameters
        ----------
        t_shift: Quantity (time)
            Amount of time by which to shift the :class:`IrregularlySampledSignal`.

        Returns
        -------
        new_sig: :class:`SpikeTrain`
            New instance of a :class:`IrregularlySampledSignal` object
            starting at t_shift later than the original :class:`IrregularlySampledSignal`
            (the original :class:`IrregularlySampledSignal` is not modified).
        """
        new_sig = deepcopy(self)

        new_sig.times += t_shift

        return new_sig

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
            raise MergeError(
                "Cannot merge these two signals as they belong to different segments.")
        if hasattr(self, "lazy_shape"):
            if hasattr(other, "lazy_shape"):
                if self.lazy_shape[0] != other.lazy_shape[0]:
                    raise MergeError("Cannot merge signals of different length.")
                merged_lazy_shape = (self.lazy_shape[0], self.lazy_shape[1] + other.lazy_shape[1])
            else:
                raise MergeError("Cannot merge a lazy object with a real object.")
        if other.units != self.units:
            other = other.rescale(self.units)
        stack = np.hstack((self.magnitude, other.magnitude))
        kwargs = {}
        for name in ("name", "description", "file_origin"):
            attr_self = getattr(self, name)
            attr_other = getattr(other, name)
            if attr_self == attr_other:
                kwargs[name] = attr_self
            else:
                kwargs[name] = "merge({}, {})".format(attr_self, attr_other)
        merged_annotations = merge_annotations(self.annotations, other.annotations)
        kwargs.update(merged_annotations)

        signal = self.__class__(self.times, stack, units=self.units, dtype=self.dtype,
                                copy=False, **kwargs)
        signal.segment = self.segment
        signal.array_annotate(**self._merge_array_annotations(other))

        if hasattr(self, "lazy_shape"):
            signal.lazy_shape = merged_lazy_shape

        return signal

    def concatenate(self, other, allow_overlap=False):
        '''
        Combine this and another signal along the time axis.

        The signal objects are concatenated vertically
        (row-wise, :func:`np.vstack`). Patching can be
        used to combine signals across segments.
        Note: Only array annotations common to
        both signals are attached to the concatenated signal.

        If the attributes of the two signal are not
        compatible, an Exception is raised.

        Required attributes of the signal are used.

        Parameters
        ----------
        other : neo.BaseSignal
            The object that is merged into this one.
        allow_overlap : bool
            If false, overlapping samples between the two
            signals are not permitted and an ValueError is raised.
            If true, no check for overlapping samples is
            performed and all samples are combined.

        Returns
        -------
        signal : neo.IrregularlySampledSignal
            Signal containing all non-overlapping samples of
            both source signals.

        Raises
        ------
        MergeError
            If `other` object has incompatible attributes.
        '''

        for attr in self._necessary_attrs:
            if not (attr[0] in ['signal', 'times', 't_start', 't_stop', 'times']):
                if getattr(self, attr[0], None) != getattr(other, attr[0], None):
                    raise MergeError(
                        "Cannot concatenate these two signals as the %s differ." % attr[0])

        if hasattr(self, "lazy_shape"):
            if hasattr(other, "lazy_shape"):
                if self.lazy_shape[-1] != other.lazy_shape[-1]:
                    raise MergeError("Cannot concatenate signals as they contain"
                                     " different numbers of traces.")
                merged_lazy_shape = (self.lazy_shape[0] + other.lazy_shape[0], self.lazy_shape[-1])
            else:
                raise MergeError("Cannot concatenate a lazy object with a real object.")
        if other.units != self.units:
            other = other.rescale(self.units)

        new_times = np.hstack((self.times, other.times))
        sorting = np.argsort(new_times)
        new_samples = np.vstack((self.magnitude, other.magnitude))

        kwargs = {}
        for name in ("name", "description", "file_origin"):
            attr_self = getattr(self, name)
            attr_other = getattr(other, name)
            if attr_self == attr_other:
                kwargs[name] = attr_self
            else:
                kwargs[name] = "merge({}, {})".format(attr_self, attr_other)
        merged_annotations = merge_annotations(self.annotations, other.annotations)
        kwargs.update(merged_annotations)

        kwargs['array_annotations'] = intersect_annotations(self.array_annotations,
                                                            other.array_annotations)

        if not allow_overlap:
            if max(self.t_start, other.t_start) <= min(self.t_stop, other.t_stop):
                raise ValueError('Can not combine signals that overlap in time. Allow for '
                                 'overlapping samples using the "no_overlap" parameter.')

        t_start = min(self.t_start, other.t_start)
        t_stop = max(self.t_start, other.t_start)

        signal = IrregularlySampledSignal(signal=new_samples[sorting], times=new_times[sorting],
                                          units=self.units, dtype=self.dtype, copy=False,
                                          t_start=t_start, t_stop=t_stop, **kwargs)
        signal.segment = None

        if hasattr(self, "lazy_shape"):
            signal.lazy_shape = merged_lazy_shape

        return signal
