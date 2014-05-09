# -*- coding: utf-8 -*-
'''
This module defines :class:`Unit`, a container of :class:`Spike` and
:class:`SpikeTrain` objects from a unit.

:class:`Unit` derives from :class:`Container`,
from :module:`neo.core.container`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np

from neo.core.container import Container


class Unit(Container):
    '''
    A container of :class:`Spike` and :class:`SpikeTrain` objects from a unit.

    A :class:`Unit` regroups all the :class:`SpikeTrain` and :class:`Spike`
    objects that were emitted by a single spike source during a :class:`Block`.
    A spike source is often a single neuron but doesn't have to be.  The spikes
    may come from different :class:`Segment` objects within the :class:`Block`,
    so this object is not contained in the usual :class:`Block`/
    :class:`Segment`/:class:`SpikeTrain` hierarchy.

    A :class:`Unit` is linked to :class:`RecordingChannelGroup` objects from
    which it was detected. With tetrodes, for instance, multiple channels may
    record the same :class:`Unit`.

    *Usage*::

        >>> from neo.core import Unit, SpikeTrain
        >>>
        >>> unit = Unit(name='pyramidal neuron')
        >>>
        >>> train0 = SpikeTrain(times=[.01, 3.3, 9.3], units='sec', t_stop=10)
        >>> unit.spiketrains.append(train0)
        >>>
        >>> train1 = SpikeTrain(times=[100.01, 103.3, 109.3], units='sec',
        ...                  t_stop=110)
        >>> unit.spiketrains.append(train1)

    *Required attributes/properties*:
        None

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :channel_index: (numpy array 1D dtype='i') You can use this to order
            :class:`Unit` objects in an way you want. It can have any number
            of elements.  :class:`AnalogSignal` and :class:`AnalogSignalArray`
            objects can be given indexes as well so related objects can be
            linked together.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    *Container of*:
        :class:`SpikeTrain`
        :class:`Spike`

    '''

    _data_child_objects = ('Spike', 'SpikeTrain')
    _single_parent_objects = ('RecordingChannelGroup',)
    _recommended_attrs = ((('channel_indexes', np.ndarray, 1, np.dtype('i')),)
                          + Container._recommended_attrs)

    def __init__(self, name=None, description=None, file_origin=None,
                 channel_indexes=None, **annotations):
        '''
        Initialize a new :clas:`Unit` instance (spike source)
        '''
        super(Unit, self).__init__(name=name, description=description,
                                   file_origin=file_origin, **annotations)

        self.channel_indexes = channel_indexes

        self.recordingchannelgroup = None
