# -*- coding: utf-8 -*-
'''
This module defines :class:`Epoch`, an array of epochs.

:class:`Epoch` derives from :class:`BaseNeo`, from
:module:`neo.core.baseneo`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, merge_annotations

PY_VER = sys.version_info[0]


class Epoch(BaseNeo, pq.Quantity):
    '''
    Array of epochs.

    *Usage*::

        >>> from neo.core import Epoch
        >>> from quantities import s, ms
        >>> import numpy as np
        >>>
        >>> epc = Epoch(times=np.arange(0, 30, 10)*s,
        ...             durations=[10, 5, 7]*ms,
        ...             labels=np.array(['btn0', 'btn1', 'btn2'], dtype='S'))
        >>>
        >>> epc.times
        array([  0.,  10.,  20.]) * s
        >>> epc.durations
        array([ 10.,   5.,   7.]) * ms
        >>> epc.labels
        array(['btn0', 'btn1', 'btn2'],
              dtype='|S4')

    *Required attributes/properties*:
        :times: (quantity array 1D) The starts of the time periods.
        :durations: (quantity array 1D) The length of the time period.
        :labels: (numpy.array 1D dtype='S') Names or labels for the
            time periods.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset,
        :description: (str) Text description,
        :file_origin: (str) Filesystem path or URL of the original data file.

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`,

    '''

    _single_parent_objects = ('Segment',)
    _quantity_attr = 'times'
    _necessary_attrs = (('times', pq.Quantity, 1),
                        ('durations', pq.Quantity, 1),
                        ('labels', np.ndarray, 1, np.dtype('S')))

    def __new__(cls, times=None, durations=None, labels=None, units=None,
                name=None, description=None, file_origin=None, **annotations):
        if times is None:
            times = np.array([]) * pq.s
        if durations is None:
            durations = np.array([]) * pq.s
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

        obj = pq.Quantity.__new__(cls, times, units=dim)
        obj.durations = durations
        obj.labels = labels
        obj.segment = None
        return obj

    def __init__(self, times=None, durations=None, labels=None, units=None,
                 name=None, description=None, file_origin=None, **annotations):
        '''
        Initialize a new :class:`Epoch` instance.
        '''
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

    def __array_finalize__(self, obj):
        super(Epoch, self).__array_finalize__(obj)
        self.durations = getattr(obj, 'durations', None)
        self.labels = getattr(obj, 'labels', None)
        self.annotations = getattr(obj, 'annotations', None)
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)
        self.segment = getattr(obj, 'segment', None)

    def __repr__(self):
        '''
        Returns a string representing the :class:`Epoch`.
        '''
        # need to convert labels to unicode for python 3 or repr is messed up
        if PY_VER == 3:
            labels = self.labels.astype('U')
        else:
            labels = self.labels

        objs = ['%s@%s for %s' % (label, time, dur) for
                label, time, dur in zip(labels, self.times, self.durations)]
        return '<Epoch: %s>' % ', '.join(objs)

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
                kwargs[name] = "merge(%s, %s)" % (attr_self, attr_other)

        merged_annotations = merge_annotations(self.annotations,
                                               other.annotations)
        kwargs.update(merged_annotations)
        return Epoch(times=times, durations=durations, labels=labels, **kwargs)

    def _copy_data_complement(self, other):
        '''
        Copy the metadata from another :class:`Epoch`.
        '''
        for attr in ("labels", "durations", "name", "file_origin",
                     "description", "annotations"):
            setattr(self, attr, getattr(other, attr, None))

    def duplicate_with_new_data(self, signal):
        '''
        Create a new :class:`Epoch` with the same metadata
        but different data (times, durations)
        '''
        new = self.__class__(times=signal)
        new._copy_data_complement(self)
        return new