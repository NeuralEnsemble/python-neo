# -*- coding: utf-8 -*-
'''
This module defines :class:`Event`, an array of events.

:class:`Event` derives from :class:`BaseNeo`, from
:module:`neo.core.baseneo`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, merge_annotations

PY_VER = sys.version_info[0]

def _new_event(cls, signal, times = None, labels=None, units=None, name=None, 
               file_origin=None, description=None,
               annotations=None):
    '''
    A function to map Event.__new__ to function that
    does not do the unit checking. This is needed for pickle to work. 
    '''
    return Event(signal=signal, times=times, labels=labels, units=units, name=name, file_origin=file_origin,
                 description=description, **annotations)

class Event(BaseNeo, pq.Quantity):
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

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`.

    '''

    _single_parent_objects = ('Segment',)
    _quantity_attr = 'times'
    _necessary_attrs = (('times', pq.Quantity, 1),
                        ('labels', np.ndarray, 1, np.dtype('S')))

    def __new__(cls, times=None, labels=None, units=None, name=None, description=None,
                file_origin=None, **annotations):
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
                 file_origin=None, **annotations):
        '''
        Initialize a new :class:`Event` instance.
        '''
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
    def __reduce__(self):
        '''
        Map the __new__ function onto _new_BaseAnalogSignal, so that pickle
        works
        '''
        return _new_event, (self.__class__, self.times, np.array(self), self.labels, self.units,
                            self.name, self.file_origin, self.description,
                            self.annotations)

    def __array_finalize__(self, obj):
        super(Event, self).__array_finalize__(obj)
        self.labels = getattr(obj, 'labels', None)
        self.annotations = getattr(obj, 'annotations', None)
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)
        self.segment = getattr(obj, 'segment', None)

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
        labels = np.hstack([self.labels, other.labels])
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
        return Event(times=times, labels=labels, **kwargs)

    def _copy_data_complement(self, other):
        '''
        Copy the metadata from another :class:`Event`.
        '''
        for attr in ("labels", "name", "file_origin", "description",
                     "annotations"):
            setattr(self, attr, getattr(other, attr, None))

    def duplicate_with_new_data(self, signal):
        '''
        Create a new :class:`Event` with the same metadata
        but different data
        '''
        new = self.__class__(times=signal)
        new._copy_data_complement(self)
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