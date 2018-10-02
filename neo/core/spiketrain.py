# -*- coding: utf-8 -*-
'''
This module implements :class:`SpikeTrain`, an array of spike times.

:class:`SpikeTrain` derives from :class:`BaseNeo`, from
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

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function
import sys

import copy
import warnings

import numpy as np
import quantities as pq
from neo.core.baseneo import BaseNeo, MergeError, merge_annotations
from neo.core.dataobject import DataObject


def check_has_dimensions_time(*values):
    '''
    Verify that all arguments have a dimensionality that is compatible
    with time.
    '''
    errmsgs = []
    for value in values:
        dim = value.dimensionality
        if (len(dim) != 1 or list(dim.values())[0] != 1 or
                not isinstance(list(dim.keys())[0], pq.UnitTime)):
            errmsgs.append("value %s has dimensions %s, not [time]" %
                           (value, dim.simplified))
    if errmsgs:
        raise ValueError("\n".join(errmsgs))


def _check_time_in_range(value, t_start, t_stop, view=False):
    '''
    Verify that all times in :attr:`value` are between :attr:`t_start`
    and :attr:`t_stop` (inclusive.

    If :attr:`view` is True, vies are used for the test.
    Using drastically increases the speed, but is only safe if you are
    certain that the dtype and units are the same
    '''

    if t_start > t_stop:
        raise ValueError("t_stop (%s) is before t_start (%s)" % (t_stop, t_start))

    if not value.size:
        return

    if view:
        value = value.view(np.ndarray)
        t_start = t_start.view(np.ndarray)
        t_stop = t_stop.view(np.ndarray)

    if value.min() < t_start:
        raise ValueError("The first spike (%s) is before t_start (%s)" %
                         (value, t_start))
    if value.max() > t_stop:
        raise ValueError("The last spike (%s) is after t_stop (%s)" %
                         (value, t_stop))


def _check_waveform_dimensions(spiketrain):
    '''
    Verify that waveform is compliant with the waveform definition as
    quantity array 3D (spike, channel_index, time)
    '''

    if not spiketrain.size:
        return

    waveforms = spiketrain.waveforms

    if (waveforms is None) or (not waveforms.size):
        return

    if waveforms.shape[0] != len(spiketrain):
        raise ValueError("Spiketrain length (%s) does not match to number of "
                         "waveforms present (%s)" % (len(spiketrain),
                                                     waveforms.shape[0]))


def _new_spiketrain(cls, signal, t_stop, units=None, dtype=None,
                    copy=True, sampling_rate=1.0 * pq.Hz,
                    t_start=0.0 * pq.s, waveforms=None, left_sweep=None,
                    name=None, file_origin=None, description=None,
                    array_annotations=None, annotations=None,
                    segment=None, unit=None):
    '''
    A function to map :meth:`BaseAnalogSignal.__new__` to function that
    does not do the unit checking. This is needed for :module:`pickle` to work.
    '''
    if annotations is None:
        annotations = {}
    obj = SpikeTrain(signal, t_stop, units, dtype, copy, sampling_rate,
                     t_start, waveforms, left_sweep, name, file_origin,
                     description, array_annotations, **annotations)
    obj.segment = segment
    obj.unit = unit
    return obj


class SpikeTrain(DataObject):
    '''
    :class:`SpikeTrain` is a :class:`Quantity` array of spike times.

    It is an ensemble of action potentials (spikes) emitted by the same unit
    in a period of time.

    *Usage*::

        >>> from neo.core import SpikeTrain
        >>> from quantities import s
        >>>
        >>> train = SpikeTrain([3, 4, 5]*s, t_stop=10.0)
        >>> train2 = train[1:3]
        >>>
        >>> train.t_start
        array(0.0) * s
        >>> train.t_stop
        array(10.0) * s
        >>> train
        <SpikeTrain(array([ 3.,  4.,  5.]) * s, [0.0 s, 10.0 s])>
        >>> train2
        <SpikeTrain(array([ 4.,  5.]) * s, [0.0 s, 10.0 s])>


    *Required attributes/properties*:
        :times: (quantity array 1D, numpy array 1D, or list) The times of
            each spike.
        :units: (quantity units) Required if :attr:`times` is a list or
                :class:`~numpy.ndarray`, not if it is a
                :class:`~quantites.Quantity`.
        :t_stop: (quantity scalar, numpy scalar, or float) Time at which
            :class:`SpikeTrain` ended. This will be converted to the
            same units as :attr:`times`. This argument is required because it
            specifies the period of time over which spikes could have occurred.
            Note that :attr:`t_start` is highly recommended for the same
            reason.

    Note: If :attr:`times` contains values outside of the
    range [t_start, t_stop], an Exception is raised.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :t_start: (quantity scalar, numpy scalar, or float) Time at which
            :class:`SpikeTrain` began. This will be converted to the
            same units as :attr:`times`.
            Default: 0.0 seconds.
        :waveforms: (quantity array 3D (spike, channel_index, time))
            The waveforms of each spike.
        :sampling_rate: (quantity scalar) Number of samples per unit time
            for the waveforms.
        :left_sweep: (quantity array 1D) Time from the beginning
            of the waveform to the trigger time of the spike.
        :sort: (bool) If True, the spike train will be sorted by time.

    *Optional attributes/properties*:
        :dtype: (numpy dtype or str) Override the dtype of the signal array.
        :copy: (bool) Whether to copy the times array.  True by default.
            Must be True when you request a change of units or dtype.
        :array_annotations: (dict) Dict mapping strings to numpy arrays containing annotations \
                                   for all data points

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`.

    *Properties available on this object*:
        :sampling_period: (quantity scalar) Interval between two samples.
            (1/:attr:`sampling_rate`)
        :duration: (quantity scalar) Duration over which spikes can occur,
            read-only.
            (:attr:`t_stop` - :attr:`t_start`)
        :spike_duration: (quantity scalar) Duration of a waveform, read-only.
            (:attr:`waveform`.shape[2] * :attr:`sampling_period`)
        :right_sweep: (quantity scalar) Time from the trigger times of the
            spikes to the end of the waveforms, read-only.
            (:attr:`left_sweep` + :attr:`spike_duration`)
        :times: (quantity array 1D) Returns the :class:`SpikeTrain` as a quantity array.

    *Slicing*:
        :class:`SpikeTrain` objects can be sliced. When this occurs, a new
        :class:`SpikeTrain` (actually a view) is returned, with the same
        metadata, except that :attr:`waveforms` is also sliced in the same way
        (along dimension 0). Note that t_start and t_stop are not changed
        automatically, although you can still manually change them.

    '''

    _single_parent_objects = ('Segment', 'Unit')
    _quantity_attr = 'times'
    _necessary_attrs = (('times', pq.Quantity, 1),
                        ('t_start', pq.Quantity, 0),
                        ('t_stop', pq.Quantity, 0))
    _recommended_attrs = ((('waveforms', pq.Quantity, 3),
                           ('left_sweep', pq.Quantity, 0),
                           ('sampling_rate', pq.Quantity, 0)) +
                          BaseNeo._recommended_attrs)

    def __new__(cls, times, t_stop, units=None, dtype=None, copy=True,
                sampling_rate=1.0 * pq.Hz, t_start=0.0 * pq.s, waveforms=None,
                left_sweep=None, name=None, file_origin=None, description=None,
                array_annotations=None, **annotations):
        '''
        Constructs a new :clas:`Spiketrain` instance from data.

        This is called whenever a new :class:`SpikeTrain` is created from the
        constructor, but not when slicing.
        '''
        if len(times) != 0 and waveforms is not None and len(times) != \
                waveforms.shape[0]:
            # len(times)!=0 has been used to workaround a bug occuring during neo import
            raise ValueError(
                "the number of waveforms should be equal to the number of spikes")

        # Make sure units are consistent
        # also get the dimensionality now since it is much faster to feed
        # that to Quantity rather than a unit
        if units is None:
            # No keyword units, so get from `times`
            try:
                dim = times.units.dimensionality
            except AttributeError:
                raise ValueError('you must specify units')
        else:
            if hasattr(units, 'dimensionality'):
                dim = units.dimensionality
            else:
                dim = pq.quantity.validate_dimensionality(units)

            if hasattr(times, 'dimensionality'):
                if times.dimensionality.items() == dim.items():
                    units = None  # units will be taken from times, avoids copying
                else:
                    if not copy:
                        raise ValueError("cannot rescale and return view")
                    else:
                        # this is needed because of a bug in python-quantities
                        # see issue # 65 in python-quantities github
                        # remove this if it is fixed
                        times = times.rescale(dim)

        if dtype is None:
            if not hasattr(times, 'dtype'):
                dtype = np.float
        elif hasattr(times, 'dtype') and times.dtype != dtype:
            if not copy:
                raise ValueError("cannot change dtype and return view")

            # if t_start.dtype or t_stop.dtype != times.dtype != dtype,
            # _check_time_in_range can have problems, so we set the t_start
            # and t_stop dtypes to be the same as times before converting them
            # to dtype below
            # see ticket #38
            if hasattr(t_start, 'dtype') and t_start.dtype != times.dtype:
                t_start = t_start.astype(times.dtype)
            if hasattr(t_stop, 'dtype') and t_stop.dtype != times.dtype:
                t_stop = t_stop.astype(times.dtype)

        # check to make sure the units are time
        # this approach is orders of magnitude faster than comparing the
        # reference dimensionality
        if (len(dim) != 1 or list(dim.values())[0] != 1 or
                not isinstance(list(dim.keys())[0], pq.UnitTime)):
            ValueError("Unit has dimensions %s, not [time]" % dim.simplified)

        # Construct Quantity from data
        obj = pq.Quantity(times, units=units, dtype=dtype, copy=copy).view(cls)

        # if the dtype and units match, just copy the values here instead
        # of doing the much more expensive creation of a new Quantity
        # using items() is orders of magnitude faster
        if (hasattr(t_start, 'dtype') and t_start.dtype == obj.dtype and
                hasattr(t_start, 'dimensionality') and
                t_start.dimensionality.items() == dim.items()):
            obj.t_start = t_start.copy()
        else:
            obj.t_start = pq.Quantity(t_start, units=dim, dtype=obj.dtype)

        if (hasattr(t_stop, 'dtype') and t_stop.dtype == obj.dtype and
                hasattr(t_stop, 'dimensionality') and
                t_stop.dimensionality.items() == dim.items()):
            obj.t_stop = t_stop.copy()
        else:
            obj.t_stop = pq.Quantity(t_stop, units=dim, dtype=obj.dtype)

        # Store attributes
        obj.waveforms = waveforms
        obj.left_sweep = left_sweep
        obj.sampling_rate = sampling_rate

        # parents
        obj.segment = None
        obj.unit = None

        # Error checking (do earlier?)
        _check_time_in_range(obj, obj.t_start, obj.t_stop, view=True)

        return obj

    def __init__(self, times, t_stop, units=None, dtype=np.float,
                 copy=True, sampling_rate=1.0 * pq.Hz, t_start=0.0 * pq.s,
                 waveforms=None, left_sweep=None, name=None, file_origin=None,
                 description=None, array_annotations=None, **annotations):
        '''
        Initializes a newly constructed :class:`SpikeTrain` instance.
        '''
        # This method is only called when constructing a new SpikeTrain,
        # not when slicing or viewing. We use the same call signature
        # as __new__ for documentation purposes. Anything not in the call
        # signature is stored in annotations.

        # Calls parent __init__, which grabs universally recommended
        # attributes and sets up self.annotations
        DataObject.__init__(self, name=name, file_origin=file_origin,
                            description=description, array_annotations=array_annotations,
                            **annotations)

    def _repr_pretty_(self, pp, cycle):
        super(SpikeTrain, self)._repr_pretty_(pp, cycle)

    def rescale(self, units):
        '''
        Return a copy of the :class:`SpikeTrain` converted to the specified
        units
        '''
        obj = super(SpikeTrain, self).rescale(units)
        obj.unit = self.unit
        return obj

    def __reduce__(self):
        '''
        Map the __new__ function onto _new_BaseAnalogSignal, so that pickle
        works
        '''
        import numpy
        return _new_spiketrain, (self.__class__, numpy.array(self),
                                 self.t_stop, self.units, self.dtype, True,
                                 self.sampling_rate, self.t_start,
                                 self.waveforms, self.left_sweep,
                                 self.name, self.file_origin, self.description,
                                 self.array_annotations, self.annotations,
                                 self.segment, self.unit)

    def __array_finalize__(self, obj):
        '''
        This is called every time a new :class:`SpikeTrain` is created.

        It is the appropriate place to set default values for attributes
        for :class:`SpikeTrain` constructed by slicing or viewing.

        User-specified values are only relevant for construction from
        constructor, and these are set in __new__. Then they are just
        copied over here.

        Note that the :attr:`waveforms` attibute is not sliced here. Nor is
        :attr:`t_start` or :attr:`t_stop` modified.
        '''
        # This calls Quantity.__array_finalize__ which deals with
        # dimensionality
        super(SpikeTrain, self).__array_finalize__(obj)

        # Supposedly, during initialization from constructor, obj is supposed
        # to be None, but this never happens. It must be something to do
        # with inheritance from Quantity.
        if obj is None:
            return

        # Set all attributes of the new object `self` from the attributes
        # of `obj`. For instance, when slicing, we want to copy over the
        # attributes of the original object.
        self.t_start = getattr(obj, 't_start', None)
        self.t_stop = getattr(obj, 't_stop', None)
        self.waveforms = getattr(obj, 'waveforms', None)
        self.left_sweep = getattr(obj, 'left_sweep', None)
        self.sampling_rate = getattr(obj, 'sampling_rate', None)
        self.segment = getattr(obj, 'segment', None)
        self.unit = getattr(obj, 'unit', None)

        # The additional arguments
        self.annotations = getattr(obj, 'annotations', {})
        # Add empty array annotations, because they cannot always be copied,
        # but do not overwrite existing ones from slicing etc.
        # This ensures the attribute exists
        if not hasattr(self, 'array_annotations'):
            self.array_annotations = {}

        # Note: Array annotations have to be changed when slicing or initializing an object,
        # copying them over in spite of changed data would result in unexpected behaviour

        # Globally recommended attributes
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)

        if hasattr(obj, 'lazy_shape'):
            self.lazy_shape = obj.lazy_shape

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_st = cls(np.array(self), self.t_stop, units=self.units,
                     dtype=self.dtype, copy=True, sampling_rate=self.sampling_rate,
                     t_start=self.t_start, waveforms=self.waveforms,
                     left_sweep=self.left_sweep, name=self.name,
                     file_origin=self.file_origin, description=self.description)
        new_st.__dict__.update(self.__dict__)
        memo[id(self)] = new_st
        for k, v in self.__dict__.items():
            try:
                setattr(new_st, k, copy.deepcopy(v, memo))
            except TypeError:
                setattr(new_st, k, v)
        return new_st

    def __repr__(self):
        '''
        Returns a string representing the :class:`SpikeTrain`.
        '''
        return '<SpikeTrain(%s, [%s, %s])>' % (
            super(SpikeTrain, self).__repr__(), self.t_start, self.t_stop)

    def sort(self):
        '''
        Sorts the :class:`SpikeTrain` and its :attr:`waveforms`, if any,
        by time.
        '''
        # sort the waveforms by the times
        sort_indices = np.argsort(self)
        if self.waveforms is not None and self.waveforms.any():
            self.waveforms = self.waveforms[sort_indices]
        self.array_annotations = copy.deepcopy(self.array_annotations_at_index(sort_indices))

        # now sort the times
        # We have sorted twice, but `self = self[sort_indices]` introduces
        # a dependency on the slicing functionality of SpikeTrain.
        super(SpikeTrain, self).sort()

    def __getslice__(self, i, j):
        '''
        Get a slice from :attr:`i` to :attr:`j`.

        Doesn't get called in Python 3, :meth:`__getitem__` is called instead
        '''
        return self.__getitem__(slice(i, j))

    def __add__(self, time):
        '''
        Shifts the time point of all spikes by adding the amount in
        :attr:`time` (:class:`Quantity`)

        Raises an exception if new time points fall outside :attr:`t_start` or
        :attr:`t_stop`
        '''
        spikes = self.view(pq.Quantity)
        check_has_dimensions_time(time)
        _check_time_in_range(spikes + time, self.t_start, self.t_stop)
        return SpikeTrain(times=spikes + time, t_stop=self.t_stop,
                          units=self.units, sampling_rate=self.sampling_rate,
                          t_start=self.t_start, waveforms=self.waveforms,
                          left_sweep=self.left_sweep, name=self.name,
                          file_origin=self.file_origin,
                          description=self.description,
                          array_annotations=copy.deepcopy(self.array_annotations),
                          **self.annotations)

    def __sub__(self, time):
        '''
        Shifts the time point of all spikes by subtracting the amount in
        :attr:`time` (:class:`Quantity`)

        Raises an exception if new time points fall outside :attr:`t_start` or
        :attr:`t_stop`
        '''
        spikes = self.view(pq.Quantity)
        check_has_dimensions_time(time)
        _check_time_in_range(spikes - time, self.t_start, self.t_stop)
        return SpikeTrain(times=spikes - time, t_stop=self.t_stop,
                          units=self.units, sampling_rate=self.sampling_rate,
                          t_start=self.t_start, waveforms=self.waveforms,
                          left_sweep=self.left_sweep, name=self.name,
                          file_origin=self.file_origin,
                          description=self.description,
                          array_annotations=copy.deepcopy(self.array_annotations),
                          **self.annotations)

    def __getitem__(self, i):
        '''
        Get the item or slice :attr:`i`.
        '''
        obj = super(SpikeTrain, self).__getitem__(i)
        if hasattr(obj, 'waveforms') and obj.waveforms is not None:
            obj.waveforms = obj.waveforms.__getitem__(i)
        try:
            obj.array_annotate(**copy.deepcopy(self.array_annotations_at_index(i)))
        except AttributeError:  # If Quantity was returned, not SpikeTrain
            pass
        return obj

    def __setitem__(self, i, value):
        '''
        Set the value the item or slice :attr:`i`.
        '''
        if not hasattr(value, "units"):
            value = pq.Quantity(value, units=self.units)
            # or should we be strict: raise ValueError("Setting a value
            # requires a quantity")?
        # check for values outside t_start, t_stop
        _check_time_in_range(value, self.t_start, self.t_stop)
        super(SpikeTrain, self).__setitem__(i, value)

    def __setslice__(self, i, j, value):
        if not hasattr(value, "units"):
            value = pq.Quantity(value, units=self.units)
        _check_time_in_range(value, self.t_start, self.t_stop)
        super(SpikeTrain, self).__setslice__(i, j, value)

    def _copy_data_complement(self, other, deep_copy=False):
        '''
        Copy the metadata from another :class:`SpikeTrain`.
        Note: Array annotations can not be copied here because length of data can change
        '''
        # Note: Array annotations cannot be copied because length of data can be changed
        # here which would cause inconsistencies
        for attr in ("left_sweep", "sampling_rate", "name", "file_origin",
                     "description", "annotations"):
            attr_value = getattr(other, attr, None)
            if deep_copy:
                attr_value = copy.deepcopy(attr_value)
            setattr(self, attr, attr_value)

    def duplicate_with_new_data(self, signal, t_start=None, t_stop=None,
                                waveforms=None, deep_copy=True, units=None):
        '''
        Create a new :class:`SpikeTrain` with the same metadata
        but different data (times, t_start, t_stop)
        Note: Array annotations can not be copied here because length of data can change
        '''
        # using previous t_start and t_stop if no values are provided
        if t_start is None:
            t_start = self.t_start
        if t_stop is None:
            t_stop = self.t_stop
        if waveforms is None:
            waveforms = self.waveforms
        if units is None:
            units = self.units
        else:
            units = pq.quantity.validate_dimensionality(units)

        new_st = self.__class__(signal, t_start=t_start, t_stop=t_stop,
                                waveforms=waveforms, units=units)
        new_st._copy_data_complement(self, deep_copy=deep_copy)

        # Note: Array annotations are not copied here, because length of data could change

        # overwriting t_start and t_stop with new values
        new_st.t_start = t_start
        new_st.t_stop = t_stop

        # consistency check
        _check_time_in_range(new_st, new_st.t_start, new_st.t_stop, view=False)
        _check_waveform_dimensions(new_st)
        return new_st

    def time_slice(self, t_start, t_stop):
        '''
        Creates a new :class:`SpikeTrain` corresponding to the time slice of
        the original :class:`SpikeTrain` between (and including) times
        :attr:`t_start` and :attr:`t_stop`. Either parameter can also be None
        to use infinite endpoints for the time interval.
        '''
        _t_start = t_start
        _t_stop = t_stop
        if t_start is None:
            _t_start = -np.inf
        if t_stop is None:
            _t_stop = np.inf
        indices = (self >= _t_start) & (self <= _t_stop)
        new_st = self[indices]

        new_st.t_start = max(_t_start, self.t_start)
        new_st.t_stop = min(_t_stop, self.t_stop)
        if self.waveforms is not None:
            new_st.waveforms = self.waveforms[indices]

        return new_st

    def merge(self, other):
        '''
        Merge another :class:`SpikeTrain` into this one.

        The times of the :class:`SpikeTrain` objects combined in one array
        and sorted.

        If the attributes of the two :class:`SpikeTrain` are not
        compatible, an Exception is raised.
        '''
        if self.sampling_rate != other.sampling_rate:
            raise MergeError("Cannot merge, different sampling rates")
        if self.t_start != other.t_start:
            raise MergeError("Cannot merge, different t_start")
        if self.t_stop != other.t_stop:
            raise MemoryError("Cannot merge, different t_stop")
        if self.left_sweep != other.left_sweep:
            raise MemoryError("Cannot merge, different left_sweep")
        if self.segment != other.segment:
            raise MergeError("Cannot merge these two signals as they belong to"
                             " different segments.")
        if hasattr(self, "lazy_shape"):
            if hasattr(other, "lazy_shape"):
                merged_lazy_shape = (self.lazy_shape[0] + other.lazy_shape[0])
            else:
                raise MergeError("Cannot merge a lazy object with a real"
                                 " object.")
        if other.units != self.units:
            other = other.rescale(self.units)
        wfs = [self.waveforms is not None, other.waveforms is not None]
        if any(wfs) and not all(wfs):
            raise MergeError("Cannot merge signal with waveform and signal "
                             "without waveform.")
        stack = np.concatenate((np.asarray(self), np.asarray(other)))
        sorting = np.argsort(stack)
        stack = stack[sorting]
        kwargs = {}

        kwargs['array_annotations'] = self._merge_array_annotations(other, sorting=sorting)

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

        train = SpikeTrain(stack, units=self.units, dtype=self.dtype,
                           copy=False, t_start=self.t_start,
                           t_stop=self.t_stop,
                           sampling_rate=self.sampling_rate,
                           left_sweep=self.left_sweep, **kwargs)
        if all(wfs):
            wfs_stack = np.vstack((self.waveforms, other.waveforms))
            wfs_stack = wfs_stack[sorting]
            train.waveforms = wfs_stack
        train.segment = self.segment
        if train.segment is not None:
            self.segment.spiketrains.append(train)

        if hasattr(self, "lazy_shape"):
            train.lazy_shape = merged_lazy_shape
        return train

    def _merge_array_annotations(self, other, sorting=None):
        '''
        Merges array annotations of 2 different objects.
        The merge happens in such a way that the result fits the merged data
        In general this means concatenating the arrays from the 2 objects.
        If an annotation is only present in one of the objects, it will be omitted.
        Apart from that the array_annotations need to be sorted according to the sorting of
        the spikes.
        :return Merged array_annotations
        '''

        assert sorting is not None, "The order of the merged spikes must be known"

        # Make sure the user is notified for every object about which exact annotations are lost
        warnings.simplefilter('always', UserWarning)

        merged_array_annotations = {}

        omitted_keys_self = []

        keys = self.array_annotations.keys()
        for key in keys:
            try:
                self_ann = copy.deepcopy(self.array_annotations[key])
                other_ann = copy.deepcopy(other.array_annotations[key])
                if isinstance(self_ann, pq.Quantity):
                    other_ann.rescale(self_ann.units)
                    arr_ann = np.concatenate([self_ann, other_ann]) * self_ann.units
                else:
                    arr_ann = np.concatenate([self_ann, other_ann])
                merged_array_annotations[key] = arr_ann[sorting]
            # Annotation only available in 'self', must be skipped
            # Ignore annotations present only in one of the SpikeTrains
            except KeyError:
                omitted_keys_self.append(key)
                continue

        omitted_keys_other = [key for key in other.array_annotations
                              if key not in self.array_annotations]

        if omitted_keys_self or omitted_keys_other:
            warnings.warn("The following array annotations were omitted, because they were only "
                          "present in one of the merged objects: {} from the one that was merged "
                          "into and {} from the one that was merged into the other".
                          format(omitted_keys_self, omitted_keys_other), UserWarning)
        # Reset warning filter to default state
        warnings.simplefilter("default")

        return merged_array_annotations

    @property
    def times(self):
        '''
        Returns the :class:`SpikeTrain` as a quantity array.
        '''
        return pq.Quantity(self)

    @property
    def duration(self):
        '''
        Duration over which spikes can occur,

        (:attr:`t_stop` - :attr:`t_start`)
        '''
        if self.t_stop is None or self.t_start is None:
            return None
        return self.t_stop - self.t_start

    @property
    def spike_duration(self):
        '''
        Duration of a waveform.

        (:attr:`waveform`.shape[2] * :attr:`sampling_period`)
        '''
        if self.waveforms is None or self.sampling_rate is None:
            return None
        return self.waveforms.shape[2] / self.sampling_rate

    @property
    def sampling_period(self):
        '''
        Interval between two samples.

        (1/:attr:`sampling_rate`)
        '''
        if self.sampling_rate is None:
            return None
        return 1.0 / self.sampling_rate

    @sampling_period.setter
    def sampling_period(self, period):
        '''
        Setter for :attr:`sampling_period`
        '''
        if period is None:
            self.sampling_rate = None
        else:
            self.sampling_rate = 1.0 / period

    @property
    def right_sweep(self):
        '''
        Time from the trigger times of the spikes to the end of the waveforms.

        (:attr:`left_sweep` + :attr:`spike_duration`)
        '''
        dur = self.spike_duration
        if self.left_sweep is None or dur is None:
            return None
        return self.left_sweep + dur
