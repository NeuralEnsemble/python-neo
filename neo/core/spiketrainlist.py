# -*- coding: utf-8 -*-
"""
This module implements :class:`SpikeTrainList`, a pseudo-list
which supports a multiplexed representation of spike trains
(all times in a single array, with a second array indicating which
neuron/channel the spike is from).
"""

import warnings
import numpy as np
import quantities as pq
from .spiketrain import SpikeTrain, normalize_times_array


def is_spiketrain_or_proxy(obj):
    return isinstance(obj, SpikeTrain) or getattr(obj, "proxy_for", None) == SpikeTrain


def unique(quantities):
    """np.unique doesn't work with a list of quantities of different scale,
    this function can be used instead."""
    # todo: add a tolerance to handle floating point discrepancies
    #       due to scaling.
    if len(quantities) > 0:
        common_units = quantities[0].units
        scaled_quantities = pq.Quantity(
            [q.rescale(common_units) for q in quantities],
            common_units)
        return np.unique(scaled_quantities)
    else:
        return quantities



class SpikeTrainList(object):
    """
    This class contains multiple spike trains, and can represent them
    either as a list of SpikeTrain objects or as a pair of arrays
    (all spike times in a single array, with a second array indicating which
    neuron/channel the spike is from).

    A SpikeTrainList object should behave like a list of SpikeTrains
    for iteration and item access. It is not intended to be used directly
    by users, but is available as the attribute `spiketrains` of Segments.

    Examples:

        # Create from list of SpikeTrain objects

        >>> stl = SpikeTrainList(items=(
        ...     SpikeTrain([0.5, 0.6, 23.6, 99.2], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
        ...     SpikeTrain([0.0007, 0.0112], units="s", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
        ...     SpikeTrain([1100, 88500], units="us", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
        ...     SpikeTrain([], units="ms", t_start=0 * pq.ms, t_stop=100.0 * pq.ms),
        ... ))
        >>> stl.multiplexed
        (array([0, 0, 0, 0, 1, 1, 2, 2]),
         array([ 0.5,  0.6, 23.6, 99.2,  0.7, 11.2,  1.1, 88.5]) * ms)

        # Create from a pair of arrays

        >>> stl = SpikeTrainList.from_spike_time_array(
        ...     np.array([0.5, 0.6, 0.7, 1.1, 11.2, 23.6, 88.5, 99.2]),
        ...     np.array([0, 0, 1, 2, 1, 0, 2, 0]),
        ...     all_channel_ids=[0, 1, 2, 3],
        ...     units='ms',
        ...     t_start=0 * pq.ms,
        ...     t_stop=100.0 * pq.ms)
        >>> list(stl)
        [<SpikeTrain(array([ 0.5,  0.6, 23.6, 99.2]) * ms, [0.0 ms, 100.0 ms])>,
         <SpikeTrain(array([ 0.7, 11.2]) * ms, [0.0 ms, 100.0 ms])>,
         <SpikeTrain(array([ 1.1, 88.5]) * ms, [0.0 ms, 100.0 ms])>,
         <SpikeTrain(array([], dtype=float64) * ms, [0.0 ms, 100.0 ms])>]

    """

    def __init__(self, items=None, segment=None):
        """Initialize self"""
        if items is None:
            self._items = items
        else:
            for item in items:
                if not is_spiketrain_or_proxy(item):
                    raise ValueError(
                        "`items` can only contain SpikeTrain objects or proxy pbjects")
            self._items = list(items)
        self._spike_time_array = None
        self._channel_id_array = None
        self._all_channel_ids = None
        self._spiketrain_metadata = {}
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
        items = self._items[i]
        if is_spiketrain_or_proxy(items):
            return items
        else:
            return SpikeTrainList(items=items)

    def __str__(self):
        """Return str(self)"""
        if self._items is None:
            if self._spike_time_array is None:
                return str([])
            else:
                return "SpikeTrainList containing {} spikes from {} neurons".format(
                    self._spike_time_array.size,
                    len(self._all_channel_ids))
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

    def _add_spiketrainlists(self, other, in_place=False):
        if self._spike_time_array is None or other._spike_time_array is None:
            # if either self or other is not storing multiplexed spike trains
            # we combine them using the list of SpikeTrains representation
            if self._items is None:
                self._spiketrains_from_array()
            if other._items is None:
                other._spiketrains_from_array()
            if in_place:
                self._items.extend(other._items)
                return self
            else:
                return self.__class__(items=self._items[:] + other._items)
        else:
            # both self and other are storing multiplexed spike trains
            # so we update the array representation
            if self._spiketrain_metadata['t_start'] != other._spiketrain_metadata['t_start']:
                raise ValueError("Incompatible t_start")
                # todo: adjust times and t_start of other to be compatible with self
            if self._spiketrain_metadata['t_stop'] != other._spiketrain_metadata['t_stop']:
                raise ValueError("Incompatible t_stop")
                # todo: adjust t_stop of self and other as necessary
            combined_spike_time_array = np.hstack(
                (self._spike_time_array, other._spike_time_array))
            combined_channel_id_array = np.hstack(
                (self._channel_id_array, other._channel_id_array))
            combined_channel_ids = set(list(self._all_channel_ids) + other._all_channel_ids)
            if len(combined_channel_ids) != (
                len(self._all_channel_ids) + len(other._all_channel_ids)
            ):
                raise ValueError("Duplicate channel ids, please rename channels before adding")
            if in_place:
                self._spike_time_array = combined_spike_time_array
                self._channel_id_array = combined_channel_id_array
                self._all_channel_ids = combined_channel_ids
                self._items = None
                return self
            else:
                return self.__class__.from_spike_time_array(
                    combined_spike_time_array,
                    combined_channel_id_array,
                    combined_channel_ids,
                    t_start=self._spiketrain_metadata['t_start'],
                    t_stop=self._spiketrain_metadata['t_stop'])

    def __add__(self, other):
        """Return self + other"""
        if isinstance(other, self.__class__):
            return self._add_spiketrainlists(other)
        elif other and is_spiketrain_or_proxy(other[0]):
            return self._add_spiketrainlists(
                self.__class__(items=other, segment=self.segment)
            )
        else:
            if self._items is None:
                self._spiketrains_from_array()
            return self._items + other

    def __iadd__(self, other):
        """Return self"""
        if isinstance(other, self.__class__):
            return self._add_spiketrainlists(other, in_place=True)
        elif other and is_spiketrain_or_proxy(other[0]):
            for obj in other:
                obj.segment = self.segment
            if self._items is None:
                self._spiketrains_from_array()
            self._items.extend(other)
            return self
        else:
            raise TypeError("Can only add a SpikeTrainList or a list of SpikeTrains in place")

    def __radd__(self, other):
        """Return other + self"""
        if isinstance(other, self.__class__):
            return other._add_spiketrainlists(self)
        elif other and is_spiketrain_or_proxy(other[0]):
            for obj in other:
                obj.segment = self.segment
            if self._items is None:
                self._spiketrains_from_array()
            self._items.extend(other)
            return self
        elif len(other) == 0:
            return self
        else:
            if self._items is None:
                self._spiketrains_from_array()
            return other + self._items

    def append(self, obj):
        """L.append(object) -> None -- append object to end"""
        if not is_spiketrain_or_proxy(obj):
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
                              all_channel_ids, t_stop, units=None,
                              t_start=0.0 * pq.s, **annotations):
        """Create a SpikeTrainList object from an array of spike times
        and an array of channel ids.

        *Required attributes/properties*:

        :spike_time_array: (quantity array 1D, numpy array 1D, or list) The times of
            all spikes.
        :channel_id_array: (numpy array 1D of dtype int) The id of the channel (e.g. the
            neuron) to which each spike belongs. This array should have the same length
            as :attr:`spike_time_array`
        :all_channel_ids: (list, tuple, or numpy array 1D containing integers) All
            channel ids. This is needed to represent channels in which there are no
            spikes.
        :units: (quantity units) Required if :attr:`spike_time_array` is not a
                :class:`~quantities.Quantity`.
        :t_stop: (quantity scalar, numpy scalar, or float) Time at which
            spike recording ended. This will be converted to the
            same units as :attr:`spike_time_array` or :attr:`units`.

        *Recommended attributes/properties*:
        :t_start: (quantity scalar, numpy scalar, or float) Time at which
            spike recording began. This will be converted to the
            same units as :attr:`spike_time_array` or :attr:`units`.
            Default: 0.0 seconds.


        *Optional attributes/properties*:
        """
        spike_time_array, dim = normalize_times_array(spike_time_array, units)
        obj = cls()
        obj._spike_time_array = spike_time_array
        obj._channel_id_array = channel_id_array
        obj._all_channel_ids = all_channel_ids
        obj._spiketrain_metadata = {
            "t_start": t_start,
            "t_stop": t_stop
        }
        for name, ann_value in annotations.items():
            if (not isinstance(ann_value, str)
                and hasattr(ann_value, "__len__")
                and len(ann_value) != len(all_channel_ids)
               ):
                raise ValueError(f"incorrect length for annotation '{name}'")
        obj._annotations = annotations
        return obj

    def _spiketrains_from_array(self):
        """Convert multiplexed spike time data into a list of SpikeTrain objects"""
        if self._spike_time_array is None:
            self._items = []
        else:
            self._items = []
            for i, channel_id in enumerate(self._all_channel_ids):
                mask = self._channel_id_array == channel_id
                times = self._spike_time_array[mask]
                spiketrain = SpikeTrain(times, **self._spiketrain_metadata)
                for name, value in self._annotations.items():
                    if (not isinstance(value, str)
                        and hasattr(value, "__len__")
                        and len(value) == len(self._all_channel_ids)
                       ):
                        spiketrain.annotate(**{name: value[i]})
                    else:
                        spiketrain.annotate(**{name: value})
                spiketrain.annotate(channel_id=channel_id)
                spiketrain.segment = self.segment
                self._items.append(spiketrain)

    @property
    def multiplexed(self):
        """Return spike trains as a pair of arrays.

        The first (plain NumPy) array contains the ids of the channels/neurons that produced
        each spike, the second (Quantity) array contains the times of the spikes.
        """
        if self._spike_time_array is None:
            # need to convert list of SpikeTrains into multiplexed spike times array
            if self._items is None:
                return np.array([]), np.array([])
            else:
                channel_ids = []
                spike_times = []
                dim = self._items[0].units.dimensionality
                for i, spiketrain in enumerate(self._items):
                    if hasattr(spiketrain, "load"):  # proxy object
                        spiketrain = spiketrain.load()
                    if spiketrain.times.dimensionality.items() == dim.items():
                        # no need to rescale
                        spike_times.append(spiketrain.times)
                    else:
                        spike_times.append(spiketrain.times.rescale(dim))
                    if ("channel_id" in spiketrain.annotations
                        and isinstance(spiketrain.annotations["channel_id"], int)
                        ):
                        ch_id = spiketrain.annotations["channel_id"]
                    else:
                        ch_id = i
                    channel_ids.append(ch_id * np.ones(spiketrain.shape, dtype=np.int64))
                self._spike_time_array = np.hstack(spike_times) * self._items[0].units
                self._channel_id_array = np.hstack(channel_ids)
        return self._channel_id_array, self._spike_time_array

    @property
    def t_start(self):
        if "t_start" in self._spiketrain_metadata:
            return self._spiketrain_metadata["t_start"]
        else:
            t_start_values = unique([item.t_start for item in self._items
                                    if isinstance(item, SpikeTrain)])  # ignore proxy objects
            if len(t_start_values) == 0:
                raise ValueError("t_start not defined for an empty spike train list")
            elif len(t_start_values) > 1:
                warnings.warn("Found multiple values of t_start, returning the earliest")
                t_start = t_start_values.min()
            else:
                t_start = t_start_values[0]
            self._spiketrain_metadata["t_start"] = t_start
            return t_start

    @property
    def t_stop(self):
        if "t_stop" in self._spiketrain_metadata:
            return self._spiketrain_metadata["t_stop"]
        else:
            t_stop_values = unique([item.t_stop for item in self._items
                                    if isinstance(item, SpikeTrain)])  # ignore proxy objects
            if len(t_stop_values) == 0:
                raise ValueError("t_stop not defined for an empty spike train list")
            elif len(t_stop_values) > 1:
                warnings.warn("Found multiple values of t_stop, returning the latest")
                t_stop = t_stop_values.max()
            else:
                t_stop = t_stop_values[0]
            self._spiketrain_metadata["t_stop"] = t_stop
            return t_stop

    @property
    def all_channel_ids(self):
        if self._all_channel_ids is None:
            self._all_channel_ids = []
            for i, spiketrain in enumerate(self._items):
                if ("channel_id" in spiketrain.annotations
                    and isinstance(spiketrain.annotations["channel_id"], int)
                   ):
                    ch_id = spiketrain.annotations["channel_id"]
                else:
                    ch_id = i
                self._all_channel_ids.append(ch_id)
        return self._all_channel_ids
