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

      * Grouping all :class:`AnalogSignal`\s and
        :class:`IrregularlySampledSignal`\s inside a :class:`Block` across
        :class:`Segment`\s;
      * Indexing a subset of the channels within an :class:`AnalogSignal` and
        :class:`IrregularlySampledSignal`\s;
      * Container of :class:`Unit`\s. Discharges of multiple neurons
        (:class:`Unit`\'s) can be seen on the same channel.

    *Usage 1* providing channel IDs across multiple :class:`Segment`::
        * Recording with 2 electrode arrays across 3 segments
        * Each array has 64 channels and is data is represented in a single
          :class:`AnalogSignal` object per electrode array
        * channel ids range from 0 to 127 with the first half covering
          electrode 0 and second half covering electrode 1

        >>> from neo.core import (Block, Segment, ChannelIndex,
        ...                       AnalogSignal)
        >>> from quantities import nA, kHz
        >>> import numpy as np
        ...
        >>> # create a Block with 3 Segment and 2 ChannelIndex objects
        >>> blk = Block()
        >>> for ind in range(3):
        ...     seg = Segment(name='segment %d' % ind, index=ind)
        ...     blk.segments.append(seg)
        ...
        >>> for ind in range(2):
        ...     channel_ids=np.arange(64)+ind
        ...     chx = ChannelIndex(name='Array probe %d' % ind,
        ...                        index=np.arange(64),
        ...                        channel_ids=channel_ids,
        ...                        channel_names=['Channel %i' % chid
        ...                                       for chid in channel_ids])
        ...     blk.channel_indexes.append(chx)
        ...
        >>> # Populate the Block with AnalogSignal objects
        >>> for seg in blk.segments:
        ...     for chx in blk.channel_indexes:
        ...         a = AnalogSignal(np.random.randn(10000, 64)*nA,
        ...                          sampling_rate=10*kHz)
        ...         # link AnalogSignal and ID providing channel_index
        ...         a.channel_index = chx
        ...         chx.analogsignals.append(a)
        ...         seg.analogsignals.append(a)

    *Usage 2* grouping channels::
        * Recording with a single probe with 8 channels, 4 of which belong to a
          Tetrode
        * Global channel IDs range from 0 to 8
        * An additional ChannelIndex is used to group subset of Tetrode channels

        >>> from neo.core import Block, ChannelIndex
        >>> import numpy as np
        >>> from quantities import mV, kHz
        ...
        >>> # Create a Block
        >>> blk = Block()
        >>> blk.segments.append(Segment())
        ...
        >>> # Create a signal with 8 channels and a ChannelIndex handling the
        >>> # channel IDs (see usage case 1)
        >>> sig = AnalogSignal(np.random.randn(1000, 8)*mV, sampling_rate=10*kHz)
        >>> chx = ChannelIndex(name='Probe 0', index=range(8),
        ...                    channel_ids=range(8),
        ...                    channel_names=['Channel %i' % chid
        ...                                   for chid in range(8)])
        >>> chx.analogsignals.append(sig)
        >>> sig.channel_index=chx
        >>> blk.segments[0].analogsignals.append(sig)
        ...
        >>> # Create a new ChannelIndex which groups four channels from the
        >>> # analogsignal and provides a second ID scheme
        >>> chx = ChannelIndex(name='Tetrode 0',
        ...                    channel_names=np.array(['Tetrode ch1',
        ...                                            'Tetrode ch4',
        ...                                            'Tetrode ch6',
        ...                                            'Tetrode ch7']),
        ...                    index=np.array([0, 3, 5, 6]))
        >>> # Attach the ChannelIndex to the the Block,
        >>> # but not the to the AnalogSignal, since sig.channel_index is
        >>> # already linked to the global ChannelIndex of Probe 0 created above
        >>> chx.analogsignals.append(sig)
        >>> blk.channel_indexes.append(chx)

    *Usage 3* dealing with :class:`Unit` objects::
        * Group 5 unit objects in a single :class:`ChannelIndex` object

        >>> from neo.core import Block, ChannelIndex, Unit
        ...
        >>> # Create a Block
        >>> blk = Block()
        ...
        >>> # Create a new ChannelIndex and add it to the Block
        >>> chx = ChannelIndex(index=None, name='octotrode A')
        >>> blk.channel_indexes.append(chx)
        ...
        >>> # create several Unit objects and add them to the
        >>> # ChannelIndex
        >>> for ind in range(5):
        ...     unit = Unit(name = 'unit %d' % ind,
        ...                 description='after a long and hard spike sorting')
        ...     chx.units.append(unit)

    *Required attributes/properties*:
        :index: (numpy.array 1D dtype='i')
            Index of each channel in the attached signals (AnalogSignals and
            IrregularlySampledSignals). The order of the channel IDs needs to
            be consistent across attached signals.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :channel_names: (numpy.array 1D dtype='S')
            Names for each recording channel.
        :channel_ids: (numpy.array 1D dtype='int')
            IDs of the corresponding channels referenced by 'index'.
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
        self.channel_names = np.array(channel_names)
        self.channel_ids = np.array(channel_ids)
        self.index = np.array(index)
        self.coordinates = coordinates

    def __getitem__(self, i):
        '''
        Get the item or slice :attr:`i`.
        '''
        index = self.index.__getitem__(i)
        if self.channel_names.size > 0:
            channel_names = self.channel_names[index]
            if not channel_names.shape:
                channel_names = [channel_names]
        else:
            channel_names = None
        if self.channel_ids.size > 0:
            channel_ids = self.channel_ids[index]
            if not channel_ids.shape:
                channel_ids = [channel_ids]
        else:
            channel_ids = None
        obj = ChannelIndex(index=np.arange(index.size),
                           channel_names=channel_names,
                           channel_ids=channel_ids)
        obj.block = self.block
        obj.analogsignals = self.analogsignals
        obj.irregularlysampledsignals = self.irregularlysampledsignals
        # we do not copy the list of units, since these are related to
        # the entire set of channels in the parent ChannelIndex
        return obj
