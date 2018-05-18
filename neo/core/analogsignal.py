# -*- coding: utf-8 -*-
'''
This module implements :class:`AnalogSignal`, an array of analog signals.

:class:`AnalogSignal` inherits from :class:`basesignal.BaseSignal` which 
derives from :class:`BaseNeo`, and from :class:`quantites.Quantity`which 
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

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, MergeError, merge_annotations
from neo.core.dataobject import DataObject
from neo.core.channelindex import ChannelIndex
from copy import copy, deepcopy

logger = logging.getLogger("Neo")

from neo.core.basesignal import BaseSignal


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
                           t_start=0 * pq.s, sampling_rate=None,
                           sampling_period=None, name=None, file_origin=None,
                           description=None, array_annotations=None, annotations=None,
                           channel_index=None, segment=None):
    '''
    A function to map AnalogSignal.__new__ to function that
        does not do the unit checking. This is needed for pickle to work.
    '''
    obj = cls(signal=signal, units=units, dtype=dtype, copy=copy,
              t_start=t_start, sampling_rate=sampling_rate,
              sampling_period=sampling_period, name=name,
              file_origin=file_origin, description=description,
              array_annotations=array_annotations, **annotations)
    obj.channel_index = channel_index
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
        Note that slicing an :class:`AnalogSignal` may give a different
        result to slicing the underlying NumPy array since signals
        are always two-dimensional.

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
                name=None, file_origin=None, description=None, array_annotations=None,
                **annotations):
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
        obj.channel_index = None
        return obj

    def __init__(self, signal, units=None, dtype=None, copy=True,
                 t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                 name=None, file_origin=None, description=None, array_annotations=None,
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
        DataObject.__init__(self, name=name, file_origin=file_origin,
                            description=description, array_annotations=array_annotations,
                            **annotations)

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
                                        self.array_annotations,
                                        self.annotations,
                                        self.channel_index,
                                        self.segment)

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

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_signal = cls(np.array(self), units=self.units, dtype=self.dtype,
                         t_start=self.t_start, sampling_rate=self.sampling_rate,
                         sampling_period=self.sampling_period, name=self.name,
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

    def __getitem__(self, i):   # TODO: IN BASESIGNAL, ARRAYANNOTATIONS RICHTIG SLICEN
        '''
        Get the item or slice :attr:`i`.
        '''
        obj = super(AnalogSignal, self).__getitem__(i)
        if isinstance(i, (int, np.integer)):  # a single point in time across all channels
            obj = pq.Quantity(obj.magnitude, units=obj.units)   # TODO: Should this be a quantity???
        elif isinstance(i, tuple):
            j, k = i
            if isinstance(j, (int, np.integer)):  # extract a quantity array
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
                if isinstance(k, (int, np.integer)):
                    obj = obj.reshape(-1, 1)
                if self.channel_index:
                    obj.channel_index = self.channel_index.__getitem__(k)
                obj.array_annotations = self.array_annotations_at_index(k)
        elif isinstance(i, slice):
            if i.start:
                obj.t_start = self.t_start + i.start * self.sampling_period
            obj.array_annotations = self.array_annotations
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

    def __eq__(self, other):
        '''
        Equality test (==)
        '''
        if (self.t_start != other.t_start or
                    self.sampling_rate != other.sampling_rate):
            return False
        return super(AnalogSignal, self).__eq__(other)

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

    def time_index(self, t):
        """Return the array index corresponding to the time `t`"""
        t = t.rescale(self.sampling_period.units)
        i = (t - self.t_start) / self.sampling_period
        i = int(np.rint(i.magnitude))
        return i

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
            i = self.time_index(t_start)

        # checking stop time and transforming to stop index
        if t_stop is None:
            j = len(self)
        else:
            j = self.time_index(t_stop)

        if (i < 0) or (j > len(self)):
            raise ValueError('t_start, t_stop have to be withing the analog \
                              signal duration')

        # we're going to send the list of indicies so that we get *copy* of the
        # sliced data
        obj = super(AnalogSignal, self).__getitem__(np.arange(i, j, 1))
        obj.t_start = self.t_start + i * self.sampling_period

        return obj

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
            new_signal.channel_index = None
            new_signal[i:j, :] = signal
            return new_signal
        else:
            self[i:j, :] = signal
            return self
