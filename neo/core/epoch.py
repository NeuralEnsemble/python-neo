'''
This module defines :class:`Epoch`, an array of epochs.

:class:`Epoch` derives from :class:`BaseNeo`, from
:module:`neo.core.baseneo`.
'''

from copy import deepcopy, copy
from numbers import Number

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, merge_annotations
from neo.core.dataobject import DataObject, ArrayDict


def _new_epoch(cls, times=None, durations=None, labels=None, units=None, name=None,
               description=None, file_origin=None, array_annotations=None, annotations=None,
               segment=None):
    '''
    A function to map epoch.__new__ to function that
    does not do the unit checking. This is needed for pickle to work.
    '''
    e = Epoch(times=times, durations=durations, labels=labels, units=units, name=name,
              file_origin=file_origin, description=description,
              array_annotations=array_annotations, **annotations)
    e.segment = segment
    return e


class Epoch(DataObject):
    '''
    Array of epochs.

    *Usage*::

        >>> from neo.core import Epoch
        >>> from quantities import s, ms
        >>> import numpy as np
        >>>
        >>> epc = Epoch(times=np.arange(0, 30, 10)*s,
        ...             durations=[10, 5, 7]*ms,
        ...             labels=np.array(['btn0', 'btn1', 'btn2'], dtype='U'))
        >>>
        >>> epc.times
        array([  0.,  10.,  20.]) * s
        >>> epc.durations
        array([ 10.,   5.,   7.]) * ms
        >>> epc.labels
        array(['btn0', 'btn1', 'btn2'],
              dtype='<U4')

    *Required attributes/properties*:
        :times: (quantity array 1D, numpy array 1D or list) The start times
           of each time period.
        :durations: (quantity array 1D, numpy array 1D, list, quantity scalar or float)
           The length(s) of each time period.
           If a scalar/float, the same value is used for all time periods.
        :labels: (numpy.array 1D dtype='U' or list) Names or labels for the time periods.
        :units: (quantity units or str) Required if the times is a list or NumPy
                array, not if it is a :class:`Quantity`

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset,
        :description: (str) Text description,
        :file_origin: (str) Filesystem path or URL of the original data file.

    *Optional attributes/properties*:
        :array_annotations: (dict) Dict mapping strings to numpy arrays containing annotations \
                                   for all data points

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`,

    '''

    _parent_objects = ('Segment',)
    _parent_attrs = ('segment',)
    _quantity_attr = 'times'
    _necessary_attrs = (('times', pq.Quantity, 1), ('durations', pq.Quantity, 1),
                        ('labels', np.ndarray, 1, np.dtype('U')))

    def __new__(cls, times=None, durations=None, labels=None, units=None, name=None,
                description=None, file_origin=None, array_annotations=None, **annotations):
        if times is None:
            times = np.array([]) * pq.s
        elif isinstance(times, (list, tuple)):
            times = np.array(times)
        if len(times.shape) > 1:
            raise ValueError("Times array has more than 1 dimension")
        if isinstance(durations, (list, tuple)):
            durations = np.array(durations)
        elif durations is None:
            durations = np.array([]) * pq.s
        elif isinstance(durations, Number):
            durations = durations * np.ones(times.shape)
        elif durations.size != times.size:
            if durations.size == 1:
                durations = durations * np.ones_like(times.magnitude)
            else:
                raise ValueError("Durations array has different length to times")
        if labels is None:
            labels = np.array([], dtype='U')
        else:
            labels = np.array(labels)
            if labels.size != times.size and labels.size:
                raise ValueError("Labels array has different length to times")
        if units is None:
            # No keyword units, so get from `times`
            try:
                units = times.units
                dim = units.dimensionality
            except AttributeError:
                raise ValueError('you must specify units')
        else:
            if hasattr(units, 'dimensionality'):
                dim = units.dimensionality
            else:
                dim = pq.quantity.validate_dimensionality(units)
        if not hasattr(durations, "dimensionality"):
            durations = pq.Quantity(durations, dim)
        # check to make sure the units are time
        # this approach is much faster than comparing the
        # reference dimensionality
        if (len(dim) != 1 or list(dim.values())[0] != 1 or not isinstance(list(dim.keys())[0],
                                                                          pq.UnitTime)):
            ValueError("Unit %s has dimensions %s, not [time]" % (units, dim.simplified))

        obj = pq.Quantity.__new__(cls, times, units=dim)
        obj._labels = labels
        obj._durations = durations
        obj.segment = None
        return obj

    def __init__(self, times=None, durations=None, labels=None, units=None, name=None,
                 description=None, file_origin=None, array_annotations=None, **annotations):
        '''
        Initialize a new :class:`Epoch` instance.
        '''
        DataObject.__init__(self, name=name, file_origin=file_origin, description=description,
                            array_annotations=array_annotations, **annotations)

    def __reduce__(self):
        '''
        Map the __new__ function onto _new_epoch, so that pickle
        works
        '''
        return _new_epoch, (self.__class__, self.times, self.durations, self.labels, self.units,
                            self.name, self.file_origin, self.description, self.array_annotations,
                            self.annotations, self.segment)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._durations = getattr(obj, 'durations', None)
        self._labels = getattr(obj, 'labels', None)
        self.annotations = getattr(obj, 'annotations', None)
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)
        self.segment = getattr(obj, 'segment', None)
        # Add empty array annotations, because they cannot always be copied,
        # but do not overwrite existing ones from slicing etc.
        # This ensures the attribute exists
        if not hasattr(self, 'array_annotations'):
            self.array_annotations = ArrayDict(self._get_arr_ann_length())

    def __repr__(self):
        '''
        Returns a string representing the :class:`Epoch`.
        '''

        objs = ['%s@%s for %s' % (label, str(time), str(dur)) for label, time, dur in
                zip(self.labels, self.times, self.durations)]
        return '<Epoch: %s>' % ', '.join(objs)

    def _repr_pretty_(self, pp, cycle):
        labels = ""
        if self._labels is not None:
            labels = " with labels"
        pp.text(f"{self.__class__.__name__} containing {self.size} epochs{labels}; "
        f"time units {self.units.dimensionality.string}; datatype {self.dtype} ")
        if self._has_repr_pretty_attrs_():
            pp.breakable()
            self._repr_pretty_attrs_(pp, cycle)

    def rescale(self, units, dtype=None):
        '''
        Return a copy of the :class:`Epoch` converted to the specified units
        The `dtype` argument exists only for backward compatibility within quantities, see
        https://github.com/python-quantities/python-quantities/pull/204
        :return: Copy of self with specified units
        '''
        # Use simpler functionality, if nothing will be changed
        dim = pq.quantity.validate_dimensionality(units)
        if self.dimensionality == dim:
            return self.copy()

        # Rescale the object into a new object
        obj = self.duplicate_with_new_data(
            times=self.view(pq.Quantity).rescale(dim),
            durations=self.durations.rescale(dim),
            labels=self.labels,
            units=units)

        # Expected behavior is deepcopy, so deepcopying array_annotations
        obj.array_annotations = deepcopy(self.array_annotations)
        obj.segment = self.segment
        return obj

    def __getitem__(self, i):
        '''
        Get the item or slice :attr:`i`.
        '''
        obj = super().__getitem__(i)
        obj._durations = self.durations[i]
        if self._labels is not None and self._labels.size > 0:
            obj._labels = self.labels[i]
        else:
            obj._labels = self.labels
        try:
            # Array annotations need to be sliced accordingly
            obj.array_annotate(**deepcopy(self.array_annotations_at_index(i)))
            obj._copy_data_complement(self)
        except AttributeError:  # If Quantity was returned, not Epoch
            obj.times = obj
            obj.durations = obj._durations
            obj.labels = obj._labels
        return obj

    def __getslice__(self, i, j):
        '''
        Get a slice from :attr:`i` to :attr:`j`.attr[0]

        Doesn't get called in Python 3, :meth:`__getitem__` is called instead
        '''
        return self.__getitem__(slice(i, j))

    @property
    def times(self):
        return pq.Quantity(self)

    def merge(self, other):
        '''
        Merge the another :class:`Epoch` into this one.

        The :class:`Epoch` objects are concatenated horizontally
        (column-wise), :func:`np.hstack`).

        If the attributes of the two :class:`Epoch` are not
        compatible, and Exception is raised.
        '''
        othertimes = other.times.rescale(self.times.units)
        otherdurations = other.durations.rescale(self.durations.units)
        times = np.hstack([self.times, othertimes]) * self.times.units
        durations = np.hstack([self.durations,
                               otherdurations]) * self.durations.units
        labels = np.hstack([self.labels, other.labels])
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

        kwargs['array_annotations'] = self._merge_array_annotations(other)

        return Epoch(times=times, durations=durations, labels=labels, **kwargs)

    def _copy_data_complement(self, other):
        '''
        Copy the metadata from another :class:`Epoch`.
        Note: Array annotations can not be copied here because length of data can change
        '''
        # Note: Array annotations cannot be copied because length of data could be changed
        # here which would cause inconsistencies. This is instead done locally.
        for attr in ("name", "file_origin", "description"):
            setattr(self, attr, deepcopy(getattr(other, attr, None)))
        self._copy_annotations(other)

    def _copy_annotations(self, other):
        self.annotations = deepcopy(other.annotations)

    def duplicate_with_new_data(self, times, durations, labels, units=None):
        '''
        Create a new :class:`Epoch` with the same metadata
        but different data (times, durations)

        Note: Array annotations can not be copied here because length of data can change
        '''

        if units is None:
            units = self.units
        else:
            units = pq.quantity.validate_dimensionality(units)

        new = self.__class__(times=times, durations=durations, labels=labels, units=units)
        new._copy_data_complement(self)
        new._labels = labels
        new._durations = durations
        # Note: Array annotations can not be copied here because length of data can change
        return new

    def time_slice(self, t_start, t_stop):
        '''
        Creates a new :class:`Epoch` corresponding to the time slice of
        the original :class:`Epoch` between (and including) times
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

        # Time slicing should create a deep copy of the object
        new_epc = deepcopy(self[indices])

        return new_epc

    def time_shift(self, t_shift):
        """
        Shifts an :class:`Epoch` by an amount of time.

        Parameters
        ----------
        t_shift: Quantity (time)
            Amount of time by which to shift the :class:`Epoch`.

        Returns
        -------
        epoch: :class:`Epoch`
            New instance of an :class:`Epoch` object starting at t_shift later than the
            original :class:`Epoch` (the original :class:`Epoch` is not modified).
        """
        new_epc = self.duplicate_with_new_data(times=self.times + t_shift,
                                               durations=self.durations,
                                               labels=self.labels)

        # Here we can safely copy the array annotations since we know that
        # the length of the Epoch does not change.
        new_epc.array_annotate(**self.array_annotations)

        return new_epc

    def set_labels(self, labels):
        if self.labels is not None and self.labels.size > 0 and len(labels) != self.size:
            raise ValueError("Labels array has different length to times ({} != {})"
                             .format(len(labels), self.size))
        self._labels = np.array(labels)

    def get_labels(self):
        return self._labels

    labels = property(get_labels, set_labels)

    def set_durations(self, durations):
        if self.durations is not None and self.durations.size > 0 and len(durations) != self.size:
            raise ValueError("Durations array has different length to times ({} != {})"
                             .format(len(durations), self.size))
        self._durations = durations

    def get_durations(self):
        return self._durations

    durations = property(get_durations, set_durations)
