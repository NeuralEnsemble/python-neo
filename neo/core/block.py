'''
This module defines :class:`Block`, the main container gathering all the data,
whether discrete or continuous, for a given recording session. base class
used by all :module:`neo.core` classes.

:class:`Block` derives from :class:`Container`,
from :module:`neo.core.container`.
'''

from datetime import datetime

from neo.core.container import Container, unique_objs


class Block(Container):
    '''
    Main container gathering all the data, whether discrete or continuous, for a
    given recording session.

    A block is not necessarily temporally homogeneous, in contrast to :class:`Segment`.

    *Usage*::

        >>> from neo.core import Block, Segment, Group, AnalogSignal
        >>> from quantities import nA, kHz
        >>> import numpy as np
        >>>
        >>> # create a Block with 3 Segment and 2 Group objects
        ,,, blk = Block()
        >>> for ind in range(3):
        ...     seg = Segment(name='segment %d' % ind, index=ind)
        ...     blk.segments.append(seg)
        ...
        >>> for ind in range(2):
        ...     group = Group(name='Array probe %d' % ind)
        ...     blk.groups.append(group)
        ...
        >>> # Populate the Block with AnalogSignal objects
        ... for seg in blk.segments:
        ...     for group in blk.groups:
        ...         a = AnalogSignal(np.random.randn(10000, 64)*nA,
        ...                          sampling_rate=10*kHz)
        ...         group.analogsignals.append(a)
        ...         seg.analogsignals.append(a)

    *Required attributes/properties*:
        None

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :file_datetime: (datetime) The creation date and time of the original
            data file.
        :rec_datetime: (datetime) The date and time of the original recording.

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`.

    *Container of*:
        :class:`Segment`
        :class:`Group`

    '''

    _container_child_objects = ('Segment', 'Group')
    _child_properties = ()
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
        Initialize a new :class:`Block` instance.
        '''
        super().__init__(name=name, description=description,
                                    file_origin=file_origin, **annotations)

        self.file_datetime = file_datetime
        self.rec_datetime = rec_datetime
        self.index = index
        self.regionsofinterest = []   # temporary workaround.
        # the goal is to store all sub-classes of RegionOfInterest in a single list
        # but this will need substantial changes to container handling

    @property
    def data_children_recur(self):
        '''
        All data child objects stored in the current object,
        obtained recursively.
        '''
        # subclassing this to remove duplicate objects such as SpikeTrain
        # objects in both Segment and Group
        # Only Block can have duplicate items right now, so implement
        # this here for performance reasons.
        return tuple(unique_objs(super().data_children_recur))

    def list_children_by_class(self, cls):
        '''
        List all children of a particular class recursively.

        You can either provide a class object, a class name,
        or the name of the container storing the class.
        '''
        # subclassing this to remove duplicate objects such as SpikeTrain
        # objects in both Segment and Group
        # Only Block can have duplicate items right now, so implement
        # this here for performance reasons.
        return unique_objs(super().list_children_by_class(cls))
