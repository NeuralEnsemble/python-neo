# -*- coding: utf-8 -*-
'''
This module defines :class:`Block`, the main container gathering all the data,
whether discrete or continous, for a given recording session. base class
used by all :module:`neo.core` classes.

:class:`Block` derives from :class:`Container`,
from :module:`neo.core.container`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

from datetime import datetime

from neo.core.container import Container, unique_objs


class Block(Container):
    '''
    Main container for data.

    Main container gathering all the data, whether discrete or continous, for a
    given recording session.

    A block is not necessarily temporally homogeneous, in contrast to Segment.

    *Usage*::

        >>> from neo.core import (Block, Segment, RecordingChannelGroup,
        ...                       AnalogSignalArray)
        >>> from quantities import nA, kHz
        >>> import numpy as np
        >>>
        >>> # create a Block with 3 Segment and 2 RecordingChannelGroup objects
        ,,, blk = Block()
        >>> for ind in range(3):
        ...     seg = Segment(name='segment %d' % ind, index=ind)
        ...     blk.segments.append(seg)
        ...
        >>> for ind in range(2):
        ...     rcg = RecordingChannelGroup(name='Array probe %d' % ind,
        ...                                 channel_indexes=np.arange(64))
        ...     blk.recordingchannelgroups.append(rcg)
        ...
        >>> # Populate the Block with AnalogSignalArray objects
        ... for seg in blk.segments:
        ...     for rcg in blk.recordingchannelgroups:
        ...         a = AnalogSignalArray(np.random.randn(10000, 64)*nA,
        ...                               sampling_rate=10*kHz)
        ...         rcg.analogsignalarrays.append(a)
        ...         seg.analogsignalarrays.append(a)

    *Required attributes/properties*:
        None

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :file_datetime: (datetime) The creation date and time of the original
            data file.
        :rec_datetime: (datetime) The date and time of the original recording.
        :index: (int) You can use this to define an ordering of your Block.
            It is not used by Neo in any way.

    *Properties available on this object*:
        :list_units: descends through hierarchy and returns a list of
            :class:`Unit` objects existing in the block. This shortcut exists
            because a common analysis case is analyzing all neurons that
            you recorded in a session.
        :list_recordingchannels: descends through hierarchy and returns
            a list of :class:`RecordingChannel` objects existing in the block.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    *Container of*:
        :class:`Segment`
        :class:`RecordingChannelGroup`

    '''

    _container_child_objects = ('Segment', 'RecordingChannelGroup')
    _child_properties = ('Unit', 'RecordingChannel')
    _recommended_attrs = ((('file_datetime', datetime),
                           ('rec_datetime', datetime),
                           ('index', int)) +
                          Container._recommended_attrs)
    _repr_pretty_attrs_keys_ = (Container._repr_pretty_attrs_keys_ +
                                ('file_origin', 'file_datetime',
                                 'rec_datetime', 'index'))
    _repr_pretty_containers = ('segments',)

    def __init__(self, name=None, description=None, file_origin=None,
                 file_datetime=None, rec_datetime=None, index=None,
                 **annotations):
        '''
        Initalize a new :class:`Block` instance.
        '''
        super(Block, self).__init__(name=name, description=description,
                                    file_origin=file_origin, **annotations)

        self.file_datetime = file_datetime
        self.rec_datetime = rec_datetime
        self.index = index

    @property
    def data_children_recur(self):
        '''
        All data child objects stored in the current object,
        obtained recursively.
        '''
        # subclassing this to remove duplicate objects such as SpikeTrain
        # objects in both Segment and Unit
        # Only Block can have duplicate items right now, so implement
        # this here for performance reasons.
        return tuple(unique_objs(super(Block, self).data_children_recur))

    def list_children_by_class(self, cls):
        '''
        List all children of a particular class recursively.

        You can either provide a class object, a class name,
        or the name of the container storing the class.
        '''
        # subclassing this to remove duplicate objects such as SpikeTrain
        # objects in both Segment and Unit
        # Only Block can have duplicate items right now, so implement
        # this here for performance reasons.
        return unique_objs(super(Block, self).list_children_by_class(cls))

    @property
    def list_units(self):
        '''
        Return a list of all :class:`Unit` objects in the :class:`Block`.
        '''
        return self.list_children_by_class('units')

    @property
    def list_recordingchannels(self):
        '''
        Return a list of all :class:`RecordingChannel` objects in the
        :class:`Block`.
        '''
        return self.list_children_by_class('recordingchannels')
