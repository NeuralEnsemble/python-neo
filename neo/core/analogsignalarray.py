# -*- coding: utf-8 -*-
'''
This module implements :class:`AnalogSignalArray`, an array of analog signals.

:class:`AnalogSignalArray` derives from :class:`BaseAnalogSignal`, from
:module:`neo.core.analogsignal`.

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

import logging

import numpy as np
import quantities as pq

from neo.core.analogsignal import (BaseAnalogSignal, AnalogSignal,
                                   _get_sampling_rate)
from neo.core.baseneo import BaseNeo, merge_annotations

logger = logging.getLogger("Neo")


class AnalogSignalArray(BaseAnalogSignal):
    '''
    Several continuous analog signals

    A representation of several continuous, analog signals that
    have the same duration, sampling rate and start time.
    Basically, it is a 2D array like AnalogSignal: dim 0 is time, dim 1 is
    channel index

    Inherits from :class:`quantities.Quantity`, which in turn inherits from
    :class:`numpy.ndarray`.

    *Usage*::

        >>> from neo.core import AnalogSignalArray
        >>> import quantities as pq
        >>>
        >>> sigarr = AnalogSignalArray([[1, 2, 3], [4, 5, 6]], units='V',
        ...                            sampling_rate=1*pq.Hz)
        >>>
        >>> sigarr
        <AnalogSignalArray(array([[1, 2, 3],
              [4, 5, 6]]) * mV, [0.0 s, 2.0 s], sampling rate: 1.0 Hz)>
        >>> sigarr[:,1]
        <AnalogSignal(array([2, 5]) * V, [0.0 s, 2.0 s],
            sampling rate: 1.0 Hz)>
        >>> sigarr[1, 1]
        array(5) * V

    *Required attributes/properties*:
        :signal: (quantity array 2D, numpy array 2D, or list (data, chanel))
            The data itself.
        :units: (quantity units) Required if the signal is a list or NumPy
                array, not if it is a :class:`Quantity`
        :t_start: (quantity scalar) Time when signal begins
        :sampling_rate: *or* :sampling_period: (quantity scalar) Number of
                                               samples per unit time or
                                               interval between two samples.
                                               If both are specified, they are
                                               checked for consistency.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :channel_index: (numpy array 1D dtype='i') You can use this to order
            the columns of the signal in any way you want. It should have the
            same number of elements as the signal has columns.
            :class:`AnalogSignal` and :class:`Unit` objects can be given
            indexes as well so related objects can be linked together.

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
        :channel_indexes: (numpy array 1D dtype='i') The same as
            :attr:`channel_index`, read-only.

    *Slicing*:
        :class:`AnalogSignalArray` objects can be sliced. When taking a single
        row (dimension 1, e.g. [:, 0]), a :class:`AnalogSignal` is returned.
        When taking a single element, a :class:`~quantities.Quantity` is
        returned.  Otherwise a :class:`AnalogSignalArray` (actually a view) is
        returned, with the same metadata, except that :attr:`t_start`
        is changed if the start index along dimension 1 is greater than 1.
        Getting a single item returns a :class:`~quantity.Quantity` scalar.

    *Operations available on this object*:
        == != + * /

    '''

    _single_parent_objects = ('Segment', 'RecordingChannelGroup')
    _quantity_attr = 'signal'
    _necessary_attrs = (('signal', pq.Quantity, 2),
                       ('sampling_rate', pq.Quantity, 0),
                       ('t_start', pq.Quantity, 0))
    _recommended_attrs = ((('channel_index', np.ndarray, 1, np.dtype('i')),) +
                          BaseNeo._recommended_attrs)

    def __new__(cls, signal, units=None, dtype=None, copy=True,
                t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                name=None, file_origin=None, description=None,
                channel_index=None, **annotations):
        '''
        Constructs new :class:`AnalogSignalArray` from data.

        This is called whenever a new class:`AnalogSignalArray` is created from
        the constructor, but not when slicing.
        '''
        if units is None:
            if not hasattr(signal, "units"):
                raise ValueError("Units must be specified")
        elif isinstance(signal, pq.Quantity):
            # could improve this test, what if units is a string?
            if units != signal.units:
                signal = signal.rescale(units)
        obj = pq.Quantity(signal, units=units, dtype=dtype,
                          copy=copy).view(cls)

        obj.t_start = t_start
        obj.sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)

        obj.channel_index = channel_index
        obj.segment = None
        obj.recordingchannelgroup = None

        return obj

    def __init__(self, signal, units=None, dtype=None, copy=True,
                 t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                 name=None, file_origin=None, description=None,
                 channel_index=None, **annotations):
        '''
        Initializes a newly constructed :class:`AnalogSignalArray` instance.
        '''
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

    @property
    def channel_indexes(self):
        '''
        The same as :attr:`channel_index`.
        '''
        return self.channel_index

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
        obj = super(BaseAnalogSignal, self).__getitem__(i)
        if isinstance(i, int):
            return obj
        elif isinstance(i, tuple):
            j, k = i
            if isinstance(k, int):
                if isinstance(j, slice):  # extract an AnalogSignal
                    obj = AnalogSignal(obj, sampling_rate=self.sampling_rate)
                    if j.start:
                        obj.t_start = (self.t_start +
                                       j.start * self.sampling_period)
                # return a Quantity (for some reason quantities does not
                # return a Quantity in this case)
                elif isinstance(j, int):
                    obj = pq.Quantity(obj, units=self.units)
                return obj
            elif isinstance(j, int):  # extract a quantity array
                # should be a better way to do this
                obj = pq.Quantity(np.array(obj), units=obj.units)
                return obj
            else:
                return obj
        elif isinstance(i, slice):
            if i.start:
                obj.t_start = self.t_start + i.start * self.sampling_period
            return obj
        else:
            raise IndexError("index should be an integer, tuple or slice")

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
        obj = super(BaseAnalogSignal, self).__getitem__(np.arange(i, j, 1))
        obj.t_start = self.t_start + i * self.sampling_period

        return obj

    def merge(self, other):
        '''
        Merge the another :class:`AnalogSignalArray` into this one.

        The :class:`AnalogSignalArray` objects are concatenated horizontally
        (column-wise, :func:`np.hstack`).

        If the attributes of the two :class:`AnalogSignalArray` are not
        compatible, and Exception is raised.
        '''
        assert self.sampling_rate == other.sampling_rate
        assert self.t_start == other.t_start
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
        if self.channel_index is None:
            channel_index = other.channel_index
        elif other.channel_index is None:
            channel_index = self.channel_index
        else:
            channel_index = np.append(self.channel_index,
                                      other.channel_index)
        merged_annotations = merge_annotations(self.annotations,
                                               other.annotations)
        kwargs.update(merged_annotations)
        return AnalogSignalArray(stack, units=self.units, dtype=self.dtype,
                                 copy=False, t_start=self.t_start,
                                 sampling_rate=self.sampling_rate,
                                 channel_index=channel_index,
                                 **kwargs)
