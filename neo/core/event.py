# -*- coding: utf-8 -*-
'''
This module defines :class:`Event`, an event occuring at a particular point in
time.

:class:`Event` derives from :class:`BaseNeo`, from :module:`neo.core.baseneo`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import quantities as pq

from neo.core.baseneo import BaseNeo


class Event(BaseNeo):
    '''
    An event occuring at a particular point in time.

    Useful for managing trigger, stimulus, comment, etc.

    *Usage*::

        >>> from neo.core import Event
        >>> from quantities import s
        >>>
        >>> evt = Event(50*s, label='trigger')
        >>>
        >>> evt.time
        array(50.0) * s
        >>> evt.label
        'trigger'

    *Required attributes/properties*:
        :time: (quantity scalar) The time of the event.
        :label: (str) Name or label for the event.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    '''

    _single_parent_objects = ('Segment',)
    _necessary_attrs = (('time', pq.Quantity, 0),
                       ('label', str))

    def __init__(self, time, label, name=None, description=None,
                 file_origin=None, **annotations):
        '''
        Initialize a new :class:`Event` instance.
        '''
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        self.time = time
        self.label = label

        self.segment = None

    def merge(self, other):
        '''
        Merging is not supported in :class:`Epoch`.
        '''
        raise NotImplementedError('Cannot merge Epoch objects')
