# -*- coding: utf-8 -*-
'''
This module defines :class:`Event`, an array of events.

:class:`Event` derives from :class:`BaseNeo`, from
:module:`neo.core.baseneo`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import sys
from copy import deepcopy

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, merge_annotations
from neo.core.dataobject import DataObject


PY_VER = sys.version_info[0]


def _new_event(cls, times=None, labels=None, units=None, name=None,
               file_origin=None, description=None, array_annotations=None,
               annotations=None, segment=None):
    '''
    A function to map Event.__new__ to function that
    does not do the unit checking. This is needed for pickle to work.
    '''
    e = Event(times=times, labels=labels, units=units, name=name, file_origin=file_origin,
              description=description, array_annotations=array_annotations, **annotations)
    e.segment = segment
    return e


class Event(DataObject):
    '''
    Array of events.

    *Usage*::

        >>> from neo.core import Event
        >>> from quantities import s
        >>> import numpy as np
        >>>
        >>> evt = Event(np.arange(0, 30, 10)*s,
        ...             labels=np.array(['trig0', 'trig1', 'trig2'],
        ...                             dtype='S'))
        >>>
        >>> evt.times
        array([  0.,  10.,  20.]) * s
        >>> evt.labels
        array(['trig0', 'trig1', 'trig2'],
              dtype='|S5')

    *Required attributes/properties*:
        :times: (quantity array 1D) The time of the events.
        :labels: (numpy.array 1D dtype='S') Names or labels for the events.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.

    *Optional attributes/properties*:
        :array_annotations: (dict) Dict mapping strings to numpy arrays containing annotations \
                                   for all data points

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`.

    '''

    _single_parent_objects = ('Segment',)
    _quantity_attr = 'times'
    _necessary_attrs = (('times', pq.Quantity, 1),
                        ('labels', np.ndarray, 1, np.dtype('S')))

    def __new__(cls, times=None, labels=None, units=None, name=None, description=None,
                file_origin=None, array_annotations=None, **annotations):
        if times is None:
            times = np.array([]) * pq.s
        if labels is None:
            labels = np.array([], dtype='S')
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
        # check to make sure the units are time
        # this approach is much faster than comparing the
        # reference dimensionality
        if (len(dim) != 1 or list(dim.values())[0] != 1 or
                not isinstance(list(dim.keys())[0], pq.UnitTime)):
            ValueError("Unit %s has dimensions %s, not [time]" %
                       (units, dim.simplified))

        obj = pq.Quantity(times, units=dim).view(cls)
        obj.labels = labels
        obj.segment = None
        return obj

    def __init__(self, times=None, labels=None, units=None, name=None, description=None,
                 file_origin=None, array_annotations=None, **annotations):
        '''
        Initialize a new :class:`Event` instance.
        '''
        DataObject.__init__(self, name=name, file_origin=file_origin,
                            description=description, array_annotations=array_annotations,
                            **annotations)

    def __reduce__(self):
        '''
        Map the __new__ function onto _new_event, so that pickle
        works
        '''
        return _new_event, (self.__class__, np.array(self), self.labels, self.units,
                            self.name, self.file_origin, self.description, self.array_annotations,
                            self.annotations, self.segment)

    def __array_finalize__(self, obj):
        super(Event, self).__array_finalize__(obj)
        self.annotations = getattr(obj, 'annotations', None)
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)
        self.segment = getattr(obj, 'segment', None)
        # Add empty array annotations, because they cannot always be copied,
        # but do not overwrite existing ones from slicing etc.
        # This ensures the attribute exists
        if not hasattr(self, 'array_annotations'):
            self.array_annotations = {}

    def __repr__(self):
        '''
        Returns a string representing the :class:`Event`.
        '''
        # need to convert labels to unicode for python 3 or repr is messed up
        if PY_VER == 3:
            labels = self.labels.astype('U')
        else:
            labels = self.labels
        objs = ['%s@%s' % (label, time) for label, time in zip(labels,
                                                               self.times)]
        return '<Event: %s>' % ', '.join(objs)

    def _repr_pretty_(self, pp, cycle):
        super(Event, self)._repr_pretty_(pp, cycle)

    def rescale(self, units):
        '''
        Return a copy of the :class:`Event` converted to the specified
        units
        '''
        obj = super(Event, self).rescale(units)
        obj.segment = self.segment
        return obj

    @property
    def times(self):
        return pq.Quantity(self)

    def merge(self, other):
        '''
        Merge the another :class:`Event` into this one.

        The :class:`Event` objects are concatenated horizontally
        (column-wise), :func:`np.hstack`).

        If the attributes of the two :class:`Event` are not
        compatible, and Exception is raised.
        '''
        othertimes = other.times.rescale(self.times.units)
        times = np.hstack([self.times, othertimes]) * self.times.units
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

        kwargs['array_annotations'] = self._merge_array_annotations(other)

        evt = Event(times=times, labels=kwargs['array_annotations']['labels'], **kwargs)

        return evt

    def _copy_data_complement(self, other):
        '''
        Copy the metadata from another :class:`Event`.
        Note: Array annotations can not be copied here because length of data can change
        '''
        # Note: Array annotations cannot be copied
        # because they are linked to their respective timestamps
        for attr in ("name", "file_origin", "description",
                     "annotations"):
            setattr(self, attr, getattr(other, attr, None))
        # Note: Array annotations cannot be copied because length of data can be changed
        # here which would cause inconsistencies
        # This includes labels and durations!!!

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_ev = cls(times=self.times,
                     labels=self.labels, units=self.units,
                     name=self.name, description=self.description,
                     file_origin=self.file_origin)
        new_ev.__dict__.update(self.__dict__)
        memo[id(self)] = new_ev
        for k, v in self.__dict__.items():
            try:
                setattr(new_ev, k, deepcopy(v, memo))
            except TypeError:
                setattr(new_ev, k, v)
        return new_ev

    def __getitem__(self, i):
        obj = super(Event, self).__getitem__(i)
        try:
            obj.array_annotate(**deepcopy(self.array_annotations_at_index(i)))
        except AttributeError:  # If Quantity was returned, not Event
            pass
        return obj

    def duplicate_with_new_data(self, signal, units=None):
        '''
        Create a new :class:`Event` with the same metadata
        but different data
        Note: Array annotations can not be copied here because length of data can change
        '''
        if units is None:
            units = self.units
        else:
            units = pq.quantity.validate_dimensionality(units)

        new = self.__class__(times=signal, units=units)
        new._copy_data_complement(self)
        # Note: Array annotations cannot be copied here, because length of data can be changed
        return new

    def time_slice(self, t_start, t_stop):
        '''
        Creates a new :class:`Event` corresponding to the time slice of
        the original :class:`Event` between (and including) times
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
        new_evt = self[indices]

        return new_evt

    def set_labels(self, labels):
        self.array_annotate(labels=labels)

    def get_labels(self):
        return self.array_annotations['labels']

    labels = property(get_labels, set_labels)
