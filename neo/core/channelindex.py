# -*- coding: utf-8 -*-
'''
This module defines :class:`ChannelIndex`, a container for multiple
data channels.

:class:`ChannelIndex` derives from :class:`Container`,
from :module:`neo.core.container`.
'''

# needed for Python 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import quantities as pq

from neo.core.container import Container


class ChannelIndex(Container):
    '''
    A container for indexing/grouping data channels.

    This container has several purposes:

      * Grouping all :class:`AnalogSignal`\s inside a :class:`Block`
        across :class:`Segment`\s;
      * Indexing a subset of the channels within an :class:`AnalogSignal`;
      * Container of :class:`Unit`\s. A neuron discharge (:class:`Unit`)
        can be seen by several electrodes (e.g. 4 for tetrodes).

    *Usage 1* multi :class:`Segment` recording with 2 electrode arrays::

        >>> from neo.core import (Block, Segment, ChannelIndex,
        ...                       AnalogSignal)
        >>> from quantities import nA, kHz
        >>> import numpy as np
        >>>
        >>> # create a Block with 3 Segment and 2 ChannelIndex objects
        ... blk = Block()
        >>> for ind in range(3):
        ...     seg = Segment(name='segment %d' % ind, index=ind)
        ...     blk.segments.append(seg)
        ...
        >>> for ind in range(2):
        ...     chx = ChannelIndex(name='Array probe %d' % ind,
        ...                        index=np.arange(64))
        ...     blk.channel_indexes.append(chx)
        ...
        >>> # Populate the Block with AnalogSignal objects
        ... for seg in blk.segments:
        ...     for chx in blk.channel_indexes:
        ...         a = AnalogSignal(np.random.randn(10000, 64)*nA,
        ...                          sampling_rate=10*kHz)
        ...         chx.analogsignals.append(a)
        ...         seg.analogsignals.append(a)

    *Usage 2* grouping channels::

        >>> from neo.core import Block, ChannelIndex
        >>> import numpy as np
        >>>
        >>> # Create a Block
        ... blk = Block()
        >>> blk.segments.append(Segment())
        >>>
        >>> # Create a signal with 8 channels
        ... sig = AnalogSignal(np.random.randn(1000, 8)*mV, sampling_rate=10*kHz)
        ... blk.segments[0].append(sig)
        ...
        >>> # Create a new ChannelIndex which groups three channels from the signal
        ... chx = ChannelIndex(channel_names=np.array(['ch1', 'ch4', 'ch6']),
        ...                    channel_indexes = np.array([0, 3, 5])
        >>> chx.analogsignals.append(sig)
        >>> blk.channel_indexes.append(chx)

    *Usage 3* dealing with :class:`Unit` objects::

        >>> from neo.core import Block, ChannelIndex, Unit
        >>>
        >>> # Create a Block
        >>> blk = Block()
        >>>
        >>> # Create a new ChannelIndex and add it to the Block
        >>> chx = ChannelIndex(name='octotrode A')
        >>> blk.channel_indexes.append(chx)
        >>>
        >>> # create several Unit objects and add them to the
        >>> # ChannelIndex
        ... for ind in range(5):
        ...     unit = Unit(name = 'unit %d' % ind,
        ...                 description='after a long and hard spike sorting')
        ...     chx.units.append(unit)

    *Required attributes/properties*:
        :channel_indexes: (numpy.array 1D dtype='i')
            Index of each channel in the attached signals.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :channel_names: (numpy.array 1D dtype='S')
            Names for each recording channel.
        :coordinates: (quantity array 2D (x, y, z))
            Physical or logical coordinates of all channels.

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`.

    *Container of*:
        :class:`AnalogSignal`
        :class:`IrregularlySampledSignal`
        :class:`Unit`

    '''

    _container_child_objects = ('Unit',)
    _data_child_objects = ('AnalogSignal', 'IrregularlySampledSignal')
    _single_parent_objects = ('Block',)
    _necessary_attrs = (('index', np.ndarray, 1, np.dtype('i')),)
    _recommended_attrs = ((('channel_names', np.ndarray, 1, np.dtype('S')),
                           ('channel_ids', np.ndarray, 1, np.dtype('i')),
                           ('coordinates', pq.Quantity, 2)) +
                          Container._recommended_attrs)

    def __init__(self, index, channel_names=None, channel_ids=None,
                 name=None, description=None, file_origin=None,
                 coordinates=None, **annotations):
        '''
        Initialize a new :class:`ChannelIndex` instance.
        '''
        # Inherited initialization
        # Sets universally recommended attributes, and places all others
        # in annotations
        super(ChannelIndex, self).__init__(name=name,
                                           description=description,
                                           file_origin=file_origin,
                                           **annotations)

        # Defaults
        if channel_names is None:
            channel_names = np.array([], dtype='S')
        if channel_ids is None:
            channel_ids = np.array([], dtype='i')

        # Store recommended attributes
        self.channel_names = channel_names
        self.channel_ids = channel_ids
        self.index = index
        self.coordinates = coordinates
