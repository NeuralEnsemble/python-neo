"""
This module implements :class:`Group`, which represents a subset of the
channels in an :class:`AnalogSignal` or :class:`IrregularlySampledSignal`.

It replaces and extends the grouping function of the former :class:`ChannelIndex`
and :class:`Unit`.
"""

from neo.core.container import Container
from neo.core.analogsignal import AnalogSignal
from neo.core.container import Container
from neo.core.objectlist import ObjectList
from neo.core.epoch import Epoch
from neo.core.event import Event
from neo.core.imagesequence import ImageSequence
from neo.core.irregularlysampledsignal import IrregularlySampledSignal
from neo.core.segment import Segment
from neo.core.spiketrainlist import SpikeTrainList
from neo.core.view import ChannelView
from neo.core.regionofinterest import RegionOfInterest


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
        :class:`Event`, :class:`Epoch`, :class:`ChannelView`, :class:`Group`
    """

    _data_child_objects = (
        "AnalogSignal",
        "IrregularlySampledSignal",
        "SpikeTrain",
        "Event",
        "Epoch",
        "ChannelView",
        "ImageSequence",
        "CircularRegionOfInterest",
        "RectangularRegionOfInterest",
        "PolygonRegionOfInterest",
    )
    _container_child_objects = ("Group",)
    _parent_objects = ("Block",)

    def __init__(self, objects=None, name=None, description=None, file_origin=None, allowed_types=None, **annotations):
        super().__init__(name=name, description=description, file_origin=file_origin, **annotations)

        # note that we create the ObjectLists here _without_ a parent argument
        # since objects do not have a reference to the group(s)
        # they are contained in.
        self._analogsignals = ObjectList(AnalogSignal)
        self._irregularlysampledsignals = ObjectList(IrregularlySampledSignal)
        self._spiketrains = SpikeTrainList()
        self._events = ObjectList(Event)
        self._epochs = ObjectList(Epoch)
        self._channelviews = ObjectList(ChannelView)
        self._imagesequences = ObjectList(ImageSequence)
        self._regionsofinterest = ObjectList(RegionOfInterest)
        self._segments = ObjectList(Segment)  # to remove?
        self._groups = ObjectList(Group)

        if allowed_types is None:
            self.allowed_types = None
        else:
            self.allowed_types = tuple(allowed_types)
            for type_ in self.allowed_types:
                if type_.__name__ not in self._child_objects:
                    raise TypeError(f"Groups can not contain objects of type {type_.__name__}")

        if objects:
            self.add(*objects)

    analogsignals = property(
        fget=lambda self: self._get_object_list("_analogsignals"),
        fset=lambda self, value: self._set_object_list("_analogsignals", value),
        doc="list of AnalogSignals contained in this group",
    )

    irregularlysampledsignals = property(
        fget=lambda self: self._get_object_list("_irregularlysampledsignals"),
        fset=lambda self, value: self._set_object_list("_irregularlysampledsignals", value),
        doc="list of IrregularlySignals contained in this group",
    )

    events = property(
        fget=lambda self: self._get_object_list("_events"),
        fset=lambda self, value: self._set_object_list("_events", value),
        doc="list of Events contained in this group",
    )

    epochs = property(
        fget=lambda self: self._get_object_list("_epochs"),
        fset=lambda self, value: self._set_object_list("_epochs", value),
        doc="list of Epochs contained in this group",
    )

    channelviews = property(
        fget=lambda self: self._get_object_list("_channelviews"),
        fset=lambda self, value: self._set_object_list("_channelviews", value),
        doc="list of ChannelViews contained in this group",
    )

    imagesequences = property(
        fget=lambda self: self._get_object_list("_imagesequences"),
        fset=lambda self, value: self._set_object_list("_imagesequences", value),
        doc="list of ImageSequences contained in this group",
    )

    regionsofinterest = property(
        fget=lambda self: self._get_object_list("_regionsofinterest"),
        fset=lambda self, value: self._set_object_list("_regionsofinterest", value),
        doc="list of RegionOfInterest objects contained in this group",
    )

    spiketrains = property(
        fget=lambda self: self._get_object_list("_spiketrains"),
        fset=lambda self, value: self._set_object_list("_spiketrains", value),
        doc="list of SpikeTrains contained in this group",
    )

    segments = property(
        fget=lambda self: self._get_object_list("_segments"),
        fset=lambda self, value: self._set_object_list("_segments", value),
        doc="list of Segments contained in this group",
    )

    groups = property(
        fget=lambda self: self._get_object_list("_groups"),
        fset=lambda self, value: self._set_object_list("_groups", value),
        doc="list of Groups contained in this group",
    )

    def add(self, *objects):
        """Add a new Neo object to the Group"""
        for obj in objects:
            if self.allowed_types and not isinstance(obj, self.allowed_types):
                raise TypeError(f"This Group can only contain {self.allowed_types}, " f"but not {type(obj)}")
        super().add(*objects)

    def walk(self):
        """
        Walk the tree of subgroups
        """
        yield self
        for grp in self.groups:
            yield from grp.walk()
