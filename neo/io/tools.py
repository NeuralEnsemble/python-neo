"""
Tools for IO coder:
  * Creating RecordingChannel and making links with AnalogSignals and
    SPikeTrains
"""

try:
    from collections.abc import MutableSequence
except ImportError:
    from collections import MutableSequence

import numpy as np

from neo.core import (AnalogSignal, Block,
                      Epoch, Event,
                      IrregularlySampledSignal,
                      Group, ChannelView,
                      Segment, SpikeTrain)


class LazyList(MutableSequence):
    """ An enhanced list that can load its members on demand. Behaves exactly
    like a regular list for members that are Neo objects. Each item should
    contain the information that ``load_lazy_cascade`` needs to load the
    respective object.
    """
    _container_objects = {
        Block, Segment, Group}
    _neo_objects = _container_objects.union(
        [AnalogSignal, Epoch, Event, ChannelView,
         IrregularlySampledSignal, SpikeTrain])

    def __init__(self, io, lazy, items=None):
        """
        :param io: IO instance that can load items.
        :param lazy: Lazy parameter with which the container object
            using the list was loaded.
        :param items: Optional, initial list of items.
        """
        if items is None:
            self._data = []
        else:
            self._data = items
        self._lazy = lazy
        self._io = io

    def __getitem__(self, index):
        item = self._data.__getitem__(index)
        if isinstance(index, slice):
            return LazyList(self._io, item)

        if type(item) in self._neo_objects:
            return item

        loaded = self._io.load_lazy_cascade(item, self._lazy)
        self._data[index] = loaded
        return loaded

    def __delitem__(self, index):
        self._data.__delitem__(index)

    def __len__(self):
        return self._data.__len__()

    def __setitem__(self, index, value):
        self._data.__setitem__(index, value)

    def insert(self, index, value):
        self._data.insert(index, value)

    def append(self, value):
        self._data.append(value)

    def reverse(self):
        self._data.reverse()

    def extend(self, values):
        self._data.extend(values)

    def remove(self, value):
        self._data.remove(value)

    def __str__(self):
        return '<' + self.__class__.__name__ + '>' + self._data.__str__()

    def __repr__(self):
        return '<' + self.__class__.__name__ + '>' + self._data.__repr__()
