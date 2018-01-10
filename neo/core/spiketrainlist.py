# -*- coding: utf-8 -*-
"""
This module implements :class:`SpikeTrainList`, a pseudo-list
which takes care of relationships between Neo parent-child objects.

In addition, it supports a multiplexed representation of spike trains
(all times in a single array, with a second array indicating which
neuron/channel the spike is from).
"""

import numpy as np
from .spiketrain import SpikeTrain


class SpikeTrainList(object):
    """
    docstring needed
    """

    def __init__(self, items=None, segment=None):
        """Initialize self"""
        self._items = items
        self._spike_time_array = None
        self._channel_id_array = None
        self._all_channel_ids = None
        self._spiketrain_metadata = None
        self.segment = segment

    def __iter__(self):
        """Implement iter(self)"""
        if self._items is None:
            self._spiketrains_from_array()
        for item in self._items:
            yield item

    def __getitem__(self, i):
        """x.__getitem__(y) <==> x[y]"""
        if self._items is None:
            self._spiketrains_from_array()
        return self._items[i]

    def __str__(self):
        """Return str(self)"""
        if self._items is None:
            if self._spike_time_array is None:
                return str([])
            else:
                return "SpikeTrainList containing {} spikes from {} neurons".format(
                    self._spike_time_array.size,
                    self._channel_id_array.size)
        else:
            return str(self._items)

    def __len__(self):
        """Return len(self)"""
        if self._items is None:
            if self._all_channel_ids is not None:
                return len(self._all_channel_ids)
            elif self._channel_id_array is not None:
                return np.unique(self._channel_id_array).size
            else:
                return 0
        else:
            return len(self._items)

    def __add__(self, other):
        """Return self + other"""
        if isinstance(other, self.__class__):
            if self._items is None or other._items is None:
                # todo: update self._spike_time_array, etc.
                raise NotImplementedError
            else:
                self._items.extend(other._items)
            return self
        elif other and isinstance(other[0], SpikeTrain):
            for obj in other:
                obj.segment = self.segment
            self._items.extend(other)
            return self
        else:
            return self._items + other

    def __radd__(self, other):
        """Return other + self"""
        if self._items is None:
            self._spiketrains_from_array()
        other.extend(self._items)
        return other

    def append(self, obj):
        """L.append(object) -> None -- append object to end"""
        if not isinstance(obj, SpikeTrain):
            raise ValueError("Can only append SpikeTrain objects")
        if self._items is None:
            self._spiketrains_from_array()
        obj.segment = self.segment
        self._items.append(obj)

    def extend(self, iterable):
        """L.extend(iterable) -> None -- extend list by appending elements from the iterable"""
        if self._items is None:
            self._spiketrains_from_array()
        for obj in iterable:
            obj.segment = self.segment
        self._items.extend(iterable)

    @classmethod
    def from_spike_time_array(cls, spike_time_array, channel_id_array,
                              all_channel_ids=None, units='ms',
                              t_start=None, t_stop=None):
        """Create a SpikeTrainList object from an array of spike times
        and an array of channel ids."""
        obj = cls()
        obj._spike_time_array = spike_time_array
        obj._channel_id_array = channel_id_array
        obj._all_channel_ids = all_channel_ids
        obj._spiketrain_metadata = {
            "units": units,
            "t_start": t_start,
            "t_stop": t_stop
        }
        return obj

    def _spiketrains_from_array(self):
        """Convert multiplexed spike time data into a list of SpikeTrain objects"""
        if self._spike_time_array is None:
            self._items = []
        else:
            if self._all_channel_ids is None:
                all_channel_ids = np.unique(self._channel_id_array)
            else:
                all_channel_ids = self._all_channel_ids
            for channel_id in all_channel_ids:
                mask = self._channel_id_array == channel_id
                times = self._spike_time_array[mask]
                spiketrain = SpikeTrain(times, **self._spiketrain_metadata)
                spiketrain.segment = self.segment
                self._items.append(spiketrain)

    @property
    def multiplexed(self):
        """Return spike trains as a pair of arrays.

        The first array contains the ids of the channels/neurons that produced each spike,
        the second array contains the times of the spikes.
        """
        if self._spike_time_array is None:
            # need to convert list of SpikeTrains into multiplexed spike times array
            raise NotImplementedError
        return self._channel_id_array, self._spike_time_array
