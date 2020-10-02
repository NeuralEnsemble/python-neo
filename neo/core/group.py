"""
This module implements :class:`Group`, which represents a subset of the
channels in an :class:`AnalogSignal` or :class:`IrregularlySampledSignal`.

It replaces and extends the grouping function of the former :class:`ChannelIndex`
and :class:`Unit`.
"""

from neo.core.container import Container


class Group(Container):
    """
    Can contain any of the data objects, views, or other groups,
    outside the hierarchy of the segment and block containers.
    A common use is to link the :class:`SpikeTrain` objects within a :class:`Block`,
    possibly across multiple Segments, that were emitted by the same neuron.

    *Required attributes/properties*:
        None

    *Recommended attributes/properties*:
        :objects: (Neo object) Objects with which to pre-populate the :class:`Group`
        :name: (str) A label for the group.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.

    *Optional arguments*:
        :allowed_types: (list or tuple) Types of Neo object that are allowed to be
          added to the Group. If not specified, any Neo object can be added.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    *Container of*:
        :class:`AnalogSignal`, :class:`IrregularlySampledSignal`, :class:`SpikeTrain`,
        :class:`Event`, :class:`Epoch`, :class:`ChannelView`, :class:`Group
    """
    _data_child_objects = (
        'AnalogSignal', 'IrregularlySampledSignal', 'SpikeTrain',
        'Event', 'Epoch', 'ChannelView', 'ImageSequence'
    )
    _container_child_objects = ('Segment', 'Group')
    _single_parent_objects = ('Block',)

    def __init__(self, objects=None, name=None, description=None, file_origin=None,
                 allowed_types=None, **annotations):
        super().__init__(name=name, description=description,
                         file_origin=file_origin, **annotations)
        if allowed_types is None:
            self.allowed_types = None
        else:
            self.allowed_types = tuple(allowed_types)
        if objects:
            self.add(*objects)

    @property
    def _container_lookup(self):
        return {
            cls_name: getattr(self, container_name)
            for cls_name, container_name in zip(self._child_objects, self._child_containers)
        }

    def add(self, *objects):
        """Add a new Neo object to the Group"""
        for obj in objects:
            if self.allowed_types and not isinstance(obj, self.allowed_types):
                raise TypeError("This Group can only contain {}".format(self.allowed_types))
            container = self._container_lookup[obj.__class__.__name__]
            container.append(obj)
