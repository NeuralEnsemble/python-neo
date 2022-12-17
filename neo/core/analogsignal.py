'''
This module implements :class:`AnalogSignal`, an array of analog signals.

:class:`AnalogSignal` inherits from :class:`basesignal.BaseSignal` which
derives from :class:`BaseNeo`, and from :class:`quantities.Quantity`which
in turn inherits from :class:`numpy.array`.

Inheritance from :class:`numpy.array` is explained here:
http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

In brief:
* Initialization of a new object from constructor happens in :meth:`__new__`.
This is where user-specified attributes are set.

* :meth:`__array_finalize__` is called for all new objects, including those
created by slicing. This is where attributes are copied over from
the old object.
'''

import logging

try:
    import scipy.signal
except ImportError as err:
    HAVE_SCIPY = False
else:
    HAVE_SCIPY = True

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, MergeError, merge_annotations, intersect_annotations
from neo.core.dataobject import DataObject
from copy import copy, deepcopy

from neo.core.basesignal import BaseSignal

logger = logging.getLogger("Neo")


def _get_sampling_rate(sampling_rate, sampling_period):
    '''
    Gets the sampling_rate from either the sampling_period or the
    sampling_rate, or makes sure they match if both are specified
    '''
    if sampling_period is None:
        if sampling_rate is None:
            raise ValueError("You must provide either the sampling rate or " + "sampling period")
    elif sampling_rate is None:
        sampling_rate = 1.0 / sampling_period
    elif sampling_period != 1.0 / sampling_rate:
        raise ValueError('The sampling_rate has to be 1/sampling_period')
    if not hasattr(sampling_rate, 'units'):
        raise TypeError("Sampling rate/sampling period must have units")
    return sampling_rate


def _new_AnalogSignalArray(cls, signal, units=None, dtype=None, copy=True, t_start=0 * pq.s,
                           sampling_rate=None, sampling_period=None, name=None, file_origin=None,
                           description=None, array_annotations=None, annotations=None,
                           segment=None):
    '''
    A function to map AnalogSignal.__new__ to function that
        does not do the unit checking. This is needed for pickle to work.
    '''
    obj = cls(signal=signal, units=units, dtype=dtype, copy=copy,
              t_start=t_start, sampling_rate=sampling_rate,
              sampling_period=sampling_period, name=name,
              file_origin=file_origin, description=description,
              array_annotations=array_annotations, **annotations)
    obj.segment = segment
    return obj


class AnalogSignal(BaseSignal):
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
        :array_annotations: (dict) Dict mapping strings to numpy arrays containing annotations \
                                   for all data points

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

    *Slicing*:
        :class:`AnalogSignal` objects can be sliced. When taking a single
        column (dimension 0, e.g. [0, :]) or a single element,
        a :class:`~quantities.Quantity` is returned.
        Otherwise an :class:`AnalogSignal` (actually a view) is
        returned, with the same metadata, except that :attr:`t_start`
        is changed if the start index along dimension 1 is greater than 1.
        Note that slicing an :class:`AnalogSignal` may give a different
        result to slicing the underlying NumPy array since signals
        are always two-dimensional.

    *Operations available on this object*:
        == != + * /

    '''

    _parent_objects = ('Segment',)
    _parent_attrs = ('segment',)
    _quantity_attr = 'signal'
    _necessary_attrs = (('signal', pq.Quantity, 2),
                        ('sampling_rate', pq.Quantity, 0),
                        ('t_start', pq.Quantity, 0))
    _recommended_attrs = BaseNeo._recommended_attrs

    def __new__(cls, signal, units=None, dtype=None, copy=True, t_start=0 * pq.s,
                sampling_rate=None, sampling_period=None, name=None, file_origin=None,
                description=None, array_annotations=None, **annotations):
        '''
        Constructs new :class:`AnalogSignal` from data.

        This is called whenever a new class:`AnalogSignal` is created from
        the constructor, but not when slicing.

        __array_finalize__ is called on the new object.
        '''
        signal = cls._rescale(signal, units=units)
        obj = pq.Quantity(signal, units=units, dtype=dtype, copy=copy).view(cls)

        if obj.ndim == 1:
            obj.shape = (-1, 1)

        if t_start is None:
            raise ValueError('t_start cannot be None')
        obj._t_start = t_start

        obj._sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)

        obj.segment = None
        return obj

    def __init__(self, signal, units=None, dtype=None, copy=True, t_start=0 * pq.s,
                 sampling_rate=None, sampling_period=None, name=None, file_origin=None,
                 description=None, array_annotations=None, **annotations):
        '''
        Initializes a newly constructed :class:`AnalogSignal` instance.
        '''
        # This method is only called when constructing a new AnalogSignal,
        # not when slicing or viewing. We use the same call signature
        # as __new__ for documentation purposes. Anything not in the call
        # signature is stored in annotations.

        # Calls parent __init__, which grabs universally recommended
        # attributes and sets up self.annotations
        DataObject.__init__(self, name=name, file_origin=file_origin, description=description,
                            array_annotations=array_annotations, **annotations)

    def __reduce__(self):
        '''
        Map the __new__ function onto _new_AnalogSignalArray, so that pickle
        works
        '''
        return _new_AnalogSignalArray, (self.__class__, np.array(self), self.units, self.dtype,
                                        True, self.t_start, self.sampling_rate,
                                        self.sampling_period, self.name, self.file_origin,
                                        self.description, self.array_annotations,
                                        self.annotations, self.segment)

    def _array_finalize_spec(self, obj):
        '''
        Set default values for attributes specific to :class:`AnalogSignal`.

        Common attributes are defined in
        :meth:`__array_finalize__` in :class:`basesignal.BaseSignal`),
        which is called every time a new signal is created
        and calls this method.
        '''
        self._t_start = getattr(obj, '_t_start', 0 * pq.s)
        self._sampling_rate = getattr(obj, '_sampling_rate', None)
        return obj

    def __repr__(self):
        '''
        Returns a string representing the :class:`AnalogSignal`.
        '''
        return ('<%s(%s, [%s, %s], sampling rate: %s)>' % (self.__class__.__name__,
                                                           super().__repr__(),
                                                           self.t_start, self.t_stop,
                                                           self.sampling_rate))

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
            if isinstance(j, (int, np.integer)):  # extract a quantity array
                obj = pq.Quantity(obj.magnitude, units=obj.units)
            else:
                if isinstance(j, slice):
                    if j.start:
                        obj.t_start = (self.t_start + j.start * self.sampling_period)
                    if j.step:
                        obj.sampling_period *= j.step
                elif isinstance(j, np.ndarray):
                    raise NotImplementedError(
                        "Arrays not yet supported")  # in the general case, would need to return
                    #  IrregularlySampledSignal(Array)
                else:
                    raise TypeError("%s not supported" % type(j))
                if isinstance(k, (int, np.integer)):
                    obj = obj.reshape(-1, 1)
                elif k is None:
                    # matplotlib _check_1d() calls__getitem__ with (:, None) and
                    # reacts appropriately if an IndexError or ValueError is raised
                    raise IndexError("Cannot add dimensions to an AnalogSignal")
                obj.array_annotate(**deepcopy(self.array_annotations_at_index(k)))
        elif isinstance(i, slice):
            obj = super().__getitem__(i)
            if i.start:
                obj.t_start = self.t_start + i.start * self.sampling_period
            obj.array_annotations = deepcopy(self.array_annotations)
        elif isinstance(i, np.ndarray):
            # Indexing of an AnalogSignal is only consistent if the resulting number of
            # samples is the same for each trace. The time axis for these samples is not
            # guaranteed to be continuous, so returning a Quantity instead of an AnalogSignal here.
            new_time_dims = np.sum(i, axis=0)
            if len(new_time_dims) and all(new_time_dims == new_time_dims[0]):
                obj = np.asarray(self).T.__getitem__(i.T)
                obj = obj.T.reshape(self.shape[1], -1).T
                obj = pq.Quantity(obj, units=self.units)
            else:
                raise IndexError("indexing of an AnalogSignals needs to keep the same number of "
                                 "sample for each trace contained")
        else:
            raise IndexError("index should be an integer, tuple, slice or boolean numpy array")
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
        return super().__setitem__(i, value)

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

    def __eq__(self, other):
        '''
        Equality test (==)
        '''
        if (isinstance(other, AnalogSignal) and (
                self.t_start != other.t_start or self.sampling_rate != other.sampling_rate)):
            return False
        return super().__eq__(other)

    def _check_consistency(self, other):
        '''
        Check if the attributes of another :class:`AnalogSignal`
        are compatible with this one.
        '''
        if isinstance(other, AnalogSignal):
            for attr in "t_start", "sampling_rate":
                if getattr(self, attr) != getattr(other, attr):
                    raise ValueError(
                        "Inconsistent values of %s" % attr)  # how to handle name and annotations?

    def _repr_pretty_(self, pp, cycle):
        '''
        Handle pretty-printing the :class:`AnalogSignal`.
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

        _pp("sampling rate: {}".format(self.sampling_rate))
        _pp("time: {} to {}".format(self.t_start, self.t_stop))

    def time_index(self, t):
        """Return the array index (or indices) corresponding to the time (or times) `t`"""
        i = (t - self.t_start) * self.sampling_rate
        i = np.rint(i.simplified.magnitude).astype(np.int64)
        return i

    def time_slice(self, t_start, t_stop):
        '''
        Creates a new AnalogSignal corresponding to the time slice of the
        original AnalogSignal between times t_start, t_stop. Note, that for
        numerical stability reasons if t_start does not fall exactly on
        the time bins defined by the sampling_period it will be rounded to
        the nearest sampling bin. The time bin for t_stop will be chosen to
        make the duration of the resultant signal as close as possible to
        t_stop - t_start. This means that for a given duration, the size
        of the slice will always be the same.
        '''

        # checking start time and transforming to start index
        if t_start is None:
            i = 0
            t_start = 0 * pq.s
        else:
            i = self.time_index(t_start)

        # checking stop time and transforming to stop index
        if t_stop is None:
            j = len(self)
        else:
            delta = (t_stop - t_start) * self.sampling_rate
            j = i + int(np.rint(delta.simplified.magnitude))

        if (i < 0) or (j > len(self)):
            raise ValueError('t_start, t_stop have to be within the analog \
                              signal duration')

        # Time slicing should create a deep copy of the object
        obj = deepcopy(self[i:j])

        obj.t_start = self.t_start + i * self.sampling_period

        return obj

    def time_shift(self, t_shift):
        """
        Shifts a :class:`AnalogSignal` to start at a new time.

        Parameters
        ----------
        t_shift: Quantity (time)
            Amount of time by which to shift the :class:`AnalogSignal`.

        Returns
        -------
        new_sig: :class:`AnalogSignal`
            New instance of a :class:`AnalogSignal` object starting at t_shift later than the
            original :class:`AnalogSignal` (the original :class:`AnalogSignal` is not modified).
        """
        new_sig = deepcopy(self)
        new_sig.t_start = new_sig.t_start + t_shift

        return new_sig

    def splice(self, signal, copy=False):
        """
        Replace part of the current signal by a new piece of signal.

        The new piece of signal will overwrite part of the current signal
        starting at the time given by the new piece's `t_start` attribute.

        The signal to be spliced in must have the same physical dimensions,
        sampling rate, and number of channels as the current signal and
        fit within it.

        If `copy` is False (the default), modify the current signal in place.
        If `copy` is True, return a new signal and leave the current one untouched.
        In this case, the new signal will not be linked to any parent objects.
        """
        if signal.t_start < self.t_start:
            raise ValueError("Cannot splice earlier than the start of the signal")
        if signal.t_stop > self.t_stop:
            raise ValueError("Splice extends beyond signal")
        if signal.sampling_rate != self.sampling_rate:
            raise ValueError("Sampling rates do not match")
        i = self.time_index(signal.t_start)
        j = i + signal.shape[0]
        if copy:
            new_signal = deepcopy(self)
            new_signal.segment = None
            new_signal[i:j, :] = signal
            return new_signal
        else:
            self[i:j, :] = signal
            return self

    def downsample(self, downsampling_factor, **kwargs):
        """
        Downsample the data of a signal.
        This method reduces the number of samples of the AnalogSignal to a fraction of the
        original number of samples, defined by `downsampling_factor`.
        This method is a wrapper of scipy.signal.decimate and accepts the same set of keyword
        arguments, except for specifying the axis of resampling, which is fixed to the first axis
        here.

        Parameters
        ----------
        downsampling_factor: int
            Factor used for decimation of samples. Scipy recommends to call decimate multiple times
            for downsampling factors higher than 13 when using IIR downsampling (default).

        Returns
        -------
        downsampled_signal: :class:`AnalogSignal`
            New instance of a :class:`AnalogSignal` object containing the resampled data points.
            The original :class:`AnalogSignal` is not modified.

        Notes
        -----
        For resampling the signal with a fixed number of samples, see `resample` method.
        """

        if not HAVE_SCIPY:
            raise ImportError('Decimating requires availability of scipy.signal')

        # Resampling is only permitted along the time axis (axis=0)
        if 'axis' in kwargs:
            kwargs.pop('axis')

        downsampled_data = scipy.signal.decimate(self.magnitude, downsampling_factor, axis=0,
                                                 **kwargs)
        downsampled_signal = self.duplicate_with_new_data(downsampled_data)

        # since the number of channels stays the same, we can also copy array annotations here
        downsampled_signal.array_annotations = self.array_annotations.copy()
        downsampled_signal.sampling_rate = self.sampling_rate / downsampling_factor

        return downsampled_signal

    def resample(self, sample_count, **kwargs):
        """
        Resample the data points of the signal.
        This method interpolates the signal and returns a new signal with a fixed number of
        samples defined by `sample_count`.
        This method is a wrapper of scipy.signal.resample and accepts the same set of keyword
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

        Notes
        -----
        For reducing the number of samples to a fraction of the original, see `downsample` method
        """

        if not HAVE_SCIPY:
            raise ImportError('Resampling requires availability of scipy.signal')

        # Resampling is only permitted along the time axis (axis=0)
        if 'axis' in kwargs:
            kwargs.pop('axis')
        if 't' in kwargs:
            kwargs.pop('t')

        resampled_data, resampled_times = scipy.signal.resample(self.magnitude, sample_count,
                                                                t=self.times, axis=0, **kwargs)

        resampled_signal = self.duplicate_with_new_data(resampled_data)
        resampled_signal.sampling_rate = (sample_count / self.shape[0]) * self.sampling_rate

        # since the number of channels stays the same, we can also copy array annotations here
        resampled_signal.array_annotations = self.array_annotations.copy()

        return resampled_signal

    def rectify(self, **kwargs):
        """
        Rectify the signal.
        This method rectifies the signal by taking the absolute value.
        This method is a wrapper of numpy.absolute() and accepts the same set of keyword
        arguments.

        Returns
        -------
        resampled_signal: :class:`AnalogSignal`
            New instance of a :class:`AnalogSignal` object containing the rectified data points.
            The original :class:`AnalogSignal` is not modified.

        """

        # Use numpy to get the absolute value of the signal
        rectified_data = np.absolute(self.magnitude, **kwargs)

        rectified_signal = self.duplicate_with_new_data(rectified_data)

        # the sampling rate stays constant
        rectified_signal.sampling_rate = self.sampling_rate

        # since the number of channels stays the same, we can also copy array annotations here
        rectified_signal.array_annotations = self.array_annotations.copy()

        return rectified_signal

    def concatenate(self, *signals, overwrite=False, padding=False):
        """
        Concatenate multiple neo.AnalogSignal objects across time.

        Units, sampling_rate and number of signal traces must be the same
        for all signals. Otherwise a ValueError is raised.
        Note that timestamps of concatenated signals might shift in oder to
        align the sampling times of all signals.

        Parameters
        ----------
        signals: neo.AnalogSignal objects
            AnalogSignals that will be concatenated
        overwrite : bool
            If True, samples of the earlier (lower index in `signals`)
            signals are overwritten by that of later (higher index in `signals`)
            signals.
            If False, samples of the later are overwritten by earlier signal.
            Default: False
        padding : bool, scalar quantity
            Sampling values to use as padding in case signals do not overlap.
            If False, do not apply padding. Signals have to align or
            overlap. If True, signals will be padded using
            np.NaN as pad values. If a scalar quantity is provided, this
            will be used for padding. The other signal is moved
            forward in time by maximum one sampling period to
            align the sampling times of both signals.
            Default: False

        Returns
        -------
        signal: neo.AnalogSignal
            concatenated output signal
        """

        # Sanity of inputs
        if not hasattr(signals, '__iter__'):
            raise TypeError('signals must be iterable')
        if not all([isinstance(a, AnalogSignal) for a in signals]):
            raise TypeError('Entries of anasiglist have to be of type neo.AnalogSignal')
        if len(signals) == 0:
            return self

        signals = [self] + list(signals)

        # Check required common attributes: units, sampling_rate and shape[-1]
        shared_attributes = ['units', 'sampling_rate']
        attribute_values = [tuple((getattr(anasig, attr) for attr in shared_attributes))
                            for anasig in signals]
        # add shape dimensions that do not relate to time
        attribute_values = [(attribute_values[i] + (signals[i].shape[1:],))
                            for i in range(len(signals))]
        if not all([attrs == attribute_values[0] for attrs in attribute_values]):
            raise MergeError(
                f'AnalogSignals have to share {shared_attributes} attributes to be concatenated.')
        units, sr, shape = attribute_values[0]

        # find gaps between Analogsignals
        combined_time_ranges = self._concatenate_time_ranges(
            [(s.t_start, s.t_stop) for s in signals])
        missing_time_ranges = self._invert_time_ranges(combined_time_ranges)
        if len(missing_time_ranges):
            diffs = np.diff(np.asarray(missing_time_ranges), axis=1)
        else:
            diffs = []

        if padding is False and any(diffs > signals[0].sampling_period):
            raise MergeError(f'Signals are not continuous. Can not concatenate signals with gaps. '
                             f'Please provide a padding value.')
        if padding is not False:
            logger.warning('Signals will be padded using {}.'.format(padding))
            if padding is True:
                padding = np.NaN * units
            if isinstance(padding, pq.Quantity):
                padding = padding.rescale(units).magnitude
            else:
                raise MergeError('Invalid type of padding value. Please provide a bool value '
                                 'or a quantities object.')

        t_start = min([a.t_start for a in signals])
        t_stop = max([a.t_stop for a in signals])
        n_samples = int(np.rint(((t_stop - t_start) * sr).rescale('dimensionless').magnitude))
        shape = (n_samples,) + shape

        # Collect attributes and annotations across all concatenated signals
        kwargs = {}
        common_annotations = signals[0].annotations
        common_array_annotations = signals[0].array_annotations
        for anasig in signals[1:]:
            common_annotations = intersect_annotations(common_annotations, anasig.annotations)
            common_array_annotations = intersect_annotations(common_array_annotations,
                                                             anasig.array_annotations)

        kwargs['annotations'] = common_annotations
        kwargs['array_annotations'] = common_array_annotations

        for name in ("name", "description", "file_origin"):
            attr = [getattr(s, name) for s in signals]
            if all([a == attr[0] for a in attr]):
                kwargs[name] = attr[0]
            else:
                kwargs[name] = f'concatenation ({attr})'

        conc_signal = AnalogSignal(np.full(shape=shape, fill_value=padding, dtype=signals[0].dtype),
                                   sampling_rate=sr, t_start=t_start, units=units, **kwargs)

        if not overwrite:
            signals = signals[::-1]
        while len(signals) > 0:
            conc_signal.splice(signals.pop(0), copy=False)

        return conc_signal

    def _concatenate_time_ranges(self, time_ranges):
        time_ranges = sorted(time_ranges)
        new_ranges = time_ranges[:1]
        for t_start, t_stop in time_ranges[1:]:
            # time range are non continuous -> define new range
            if t_start > new_ranges[-1][1]:
                new_ranges.append((t_start, t_stop))
            # time range is continuous -> extend time range
            elif t_stop > new_ranges[-1][1]:
                new_ranges[-1] = (new_ranges[-1][0], t_stop)
        return new_ranges

    def _invert_time_ranges(self, time_ranges):
        i = 0
        new_ranges = []
        while i < len(time_ranges) - 1:
            new_ranges.append((time_ranges[i][1], time_ranges[i + 1][0]))
            i += 1
        return new_ranges
