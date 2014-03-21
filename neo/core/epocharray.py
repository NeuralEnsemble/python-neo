# -*- coding: utf-8 -*-
'''
This module defines :class:`EpochArray`, an array of epochs. Introduced for
performance reasons.

:class:`EpochArray` derives from :class:`BaseNeo`, from
:module:`neo.core.baseneo`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, merge_annotations

PY_VER = sys.version_info[0]


class EpochArray(BaseNeo):
    '''
    Array of epochs. Introduced for performance reason.

    An :class:`EpochArray` is prefered to a list of :class:`Epoch` objects.

    *Usage*::

        >>> from neo.core import EpochArray
        >>> from quantities import s, ms
        >>> import numpy as np
        >>>
        >>> epcarr = EpochArray(times=np.arange(0, 30, 10)*s,
        ...                     durations=[10, 5, 7]*ms,
        ...                     labels=np.array(['btn0', 'btn1', 'btn2'],
        ...                                     dtype='S'))
        >>>
        >>> epcarr.times
        array([  0.,  10.,  20.]) * s
        >>> epcarr.durations
        array([ 10.,   5.,   7.]) * ms
        >>> epcarr.labels
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
    _necessary_attrs = (('times', pq.Quantity, 1),
                       ('durations', pq.Quantity, 1),
                       ('labels', np.ndarray, 1, np.dtype('S')))

    def __init__(self, times=None, durations=None, labels=None,
                 name=None, description=None, file_origin=None, **annotations):
        '''
        Initialize a new :class:`EpochArray` instance.
        '''
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)

        if times is None:
            times = np.array([]) * pq.s
        if durations is None:
            durations = np.array([]) * pq.s
        if labels is None:
            labels = np.array([], dtype='S')

        self.times = times
        self.durations = durations
        self.labels = labels

        self.segment = None

    def __repr__(self):
        '''
        Returns a string representing the :class:`EpochArray`.
        '''
        # need to convert labels to unicode for python 3 or repr is messed up
        if PY_VER == 3:
            labels = self.labels.astype('U')
        else:
            labels = self.labels

        objs = ['%s@%s for %s' % (label, time, dur) for
                label, time, dur in zip(labels, self.times, self.durations)]
        return '<EpochArray: %s>' % ', '.join(objs)

    def merge(self, other):
        '''
        Merge the another :class:`EpochArray` into this one.

        The :class:`EpochArray` objects are concatenated horizontally
        (column-wise), :func:`np.hstack`).

        If the attributes of the two :class:`EpochArray` are not
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
        return EpochArray(times=times, durations=durations, labels=labels,
                          **kwargs)
