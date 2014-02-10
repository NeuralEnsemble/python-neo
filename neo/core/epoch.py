# -*- coding: utf-8 -*-
'''
This module defines :class:`Epoch`, a period of time with a start point and
duration.

:class:`Epoch` derives from :class:`BaseNeo`, from :module:`neo.core.baseneo`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import quantities as pq

from neo.core.baseneo import BaseNeo


class Epoch(BaseNeo):
    '''
    A period of time with a start point and duration.

    Similar to :class:`Event` but with a duration.
    Useful for describing a period, the state of a subject, etc.

    *Usage*::

        >>> from neo.core import Epoch
        >>> from quantities import s, ms
        >>>
        >>> epc = Epoch(time=50*s, duration=200*ms, label='button pressed')
        >>>
        >>> epc.time
        array(50.0) * s
        >>> epc.duration
        array(200.0) * ms
        >>> epc.label
        'button pressed'

    *Required attributes/properties*:
        :time: (quantity scalar) The start of the time period.
        :duration: (quantity scalar) The length of the time period.
        :label: (str) A name or label for the time period.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    '''

    _single_parent_objects = ('Segment',)
    _necessary_attrs = (('time', pq.Quantity, 0),
                       ('duration', pq.Quantity, 0),
                       ('label', str))

    def __init__(self, time, duration, label, name=None, description=None,
                 file_origin=None, **annotations):
        '''
        Initialize a new :class:`Epoch` instance.
        '''
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        self.time = time
        self.duration = duration
        self.label = label

        self.segment = None

    def merge(self, other):
        '''
        Merging is not supported in :class:`Epoch`.
        '''
        raise NotImplementedError('Cannot merge Epoch objects')
