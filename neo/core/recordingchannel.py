# -*- coding: utf-8 -*-
'''
This module defines :class:`RecordingChannel`, a container for recordings
coming from a single data channel.

:class:`RecordingChannel` derives from :class:`Container`,
from :module:`neo.core.container`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import quantities as pq

from neo.core.container import Container


class RecordingChannel(Container):
    '''
    A container for recordings coming from a single data channel.

    A :class:`RecordingChannel` is a container for :class:`AnalogSignal` and
    :class:`IrregularlySampledSignal`objects that come from the same logical
    and/or physical channel inside a :class:`Block`.

    Note that a :class:`RecordingChannel` can belong to several
    :class:`RecordingChannelGroup` objects.

    *Usage* one :class:`Block` with 3 :class:`Segment` objects, 16
    :class:`RecordingChannel` objects, and 48 :class:`AnalogSignal` objects::

        >>> from neo.core import (Block, Segment, RecordingChannelGroup,
        ...                       RecordingChannel, AnalogSignal)
        >>> from quantities import mA, Hz
        >>> import numpy as np
        >>>
        >>> # Create a Block
        ... blk = Block()
        >>>
        >>> # Create a new RecordingChannelGroup and add it to the Block
        ... rcg = RecordingChannelGroup(name='all channels')
        >>> blk.recordingchannelgroups.append(rcg)
        >>>
        >>> # Create 3 Segment and 16 RecordingChannel objects and add them to
        ... # the Block
        ... for ind in range(3):
        ...     seg = Segment(name='segment %d' % ind, index=ind)
        ...     blk.segments.append(seg)
        ...
        >>> for ind in range(16):
        ...     chan = RecordingChannel(index=ind)
        ...     rcg.recordingchannels.append(chan)  # <- many to many
        ...                                         #    relationship
        ...     chan.recordingchannelgroups.append(rcg)  # <- many to many
        ...                                              #    relationship
        ...
        >>> # Populate the Block with AnalogSignal objects
        ... for seg in blk.segments:
        ...     for chan in rcg.recordingchannels:
        ...         sig = AnalogSignal(np.random.rand(100000)*mA,
        ...                            sampling_rate=20*Hz)
        ...         seg.analogsignals.append(sig)
        ...         chan.analogsignals.append(sig)

    *Required attributes/properties*:
        :index: (int) You can use this to order :class:`RecordingChannel`
            objects in an way you want.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :coordinate: (quantity array 1D (x, y, z)) Coordinates of the channel
            in the brain.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    *Container of*:
        :class:`AnalogSignal`
        :class:`IrregularlySampledSignal`

    '''

    _data_child_objects = ('AnalogSignal', 'IrregularlySampledSignal')
    _multi_parent_objects = ('RecordingChannelGroup',)
    _necessary_attrs = (('index', int),)
    _recommended_attrs = ((('coordinate', pq.Quantity, 1),) +
                          Container._recommended_attrs)

    def __init__(self, index=0, coordinate=None, name=None, description=None,
                 file_origin=None, **annotations):
        '''
        Initialize a new :class:`RecordingChannel` instance.
        '''
        # Inherited initialization
        # Sets universally recommended attributes, and places all others
        # in annotations
        super(RecordingChannel, self).__init__(name=name,
                                               description=description,
                                               file_origin=file_origin,
                                               **annotations)

        # Store required and recommended attributes
        self.index = index
        self.coordinate = coordinate
