# -*- coding: utf-8 -*-
'''
This module defines :class:`EventArray`, an array of events. Introduced for
performance reasons.

:class:`EventArray` derives from :class:`BaseNeo`, from
:module:`neo.core.baseneo`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, merge_annotations

PY_VER = sys.version_info[0]


class EventArray(BaseNeo):
    '''
    Array of events. Introduced for performance reasons.

    An :class:`EventArray` is prefered to a list of :class:`Event` objects.

    *Usage*::

        >>> from neo.core import EventArray
        >>> from quantities import s
        >>> import numpy as np
        >>>
        >>> evtarr = EventArray(np.arange(0, 30, 10)*s,
        ...                     labels=np.array(['trig0', 'trig1', 'trig2'],
        ...                                     dtype='S'))
        >>>
        >>> evtarr.times
        array([  0.,  10.,  20.]) * s
        >>> evtarr.labels
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
    _necessary_attrs = (('times', pq.Quantity, 1),
                       ('labels', np.ndarray, 1, np.dtype('S')))

    def __init__(self, times=None, labels=None, name=None, description=None,
                 file_origin=None, **annotations):
        '''
        Initialize a new :class:`EventArray` instance.
        '''
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        if times is None:
            times = np.array([]) * pq.s
        if labels is None:
            labels = np.array([], dtype='S')

        self.times = times
        self.labels = labels

        self.segment = None

    def __repr__(self):
        '''
        Returns a string representing the :class:`EventArray`.
        '''
        # need to convert labels to unicode for python 3 or repr is messed up
        if PY_VER == 3:
            labels = self.labels.astype('U')
        else:
            labels = self.labels
        objs = ['%s@%s' % (label, time) for label, time in zip(labels,
                                                               self.times)]
        return '<EventArray: %s>' % ', '.join(objs)

    def merge(self, other):
        '''
        Merge the another :class:`EventArray` into this one.

        The :class:`EventArray` objects are concatenated horizontally
        (column-wise), :func:`np.hstack`).

        If the attributes of the two :class:`EventArray` are not
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
        return EventArray(times=times, labels=labels, **kwargs)
