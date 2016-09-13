# -*- coding: utf-8 -*-
"""


"""

from __future__ import absolute_import

import logging
import pickle
import numpy as np
import quantities as pq
try:
    import h5py
except ImportError as err:
    HAVE_H5PY = False
else:
    HAVE_H5PY = True

from neo.core import (objectlist, Block, Segment, AnalogSignal, SpikeTrain,
                      Epoch, Event, IrregularlySampledSignal, ChannelIndex,
                      Unit)
from neo.io.baseio import BaseIO
from neo.core.baseneo import MergeError

logger = logging.getLogger('Neo')


def disjoint_groups(groups):
    """`groups` should be a list of sets"""
    groups = groups[:]  # copy, so as not to change original
    for group1 in groups:
        for group2 in groups:
            if group1 != group2:
                if group2.issubset(group1):
                    groups.remove(group2)
                elif group1.issubset(group2):
                    groups.remove(group1)
    return groups


class NeoHdf5IO(BaseIO):
    """
    Class for reading HDF5 format files created by Neo version 0.4 or earlier.

    Writing to HDF5 is not supported by this IO; we recommend using NixIO for this.
    """
    supported_objects = objectlist
    readable_objects = objectlist
    name = 'NeoHdf5 IO'
    extensions = ['h5']
    mode = 'file'
    is_readable = True
    is_writable = False

    def __init__(self, filename):
        if not HAVE_H5PY:
            raise ImportError("h5py is not available")
        BaseIO.__init__(self, filename=filename)
        self._data = h5py.File(filename, 'r')
        self.object_refs = {}
        self._lazy = False

    def read_all_blocks(self, lazy=False, cascade=True, merge_singles=True, **kargs):
        """
        Loads all blocks in the file that are attached to the root (which
        happens when they are saved with save() or write_block()).

        If `merge_singles` is True, then the IO will attempt to merge single channel
         `AnalogSignal` objects into multichannel objects, and similarly for single `Epoch`,
         `Event` and `IrregularlySampledSignal` objects.
        """
        self._lazy = lazy
        self._cascade = cascade
        self.merge_singles = merge_singles

        blocks = []
        for name, node in self._data.items():
            if "Block" in name:
                blocks.append(self._read_block(node))
        return blocks

    def read_block(self, lazy=False, cascade=True, **kargs):
        """
        Load the first block in the file.
        """
        return self.read_all_blocks(lazy=lazy, cascade=cascade)[0]

    def _read_block(self, node):
        attributes = self._get_standard_attributes(node)
        block = Block(**attributes)

        if self._cascade:
            for name, child_node in node['segments'].items():
                if "Segment" in name:
                    block.segments.append(self._read_segment(child_node, parent=block))

            if len(node['recordingchannelgroups']) > 0:
                for name, child_node in node['recordingchannelgroups'].items():
                    if "RecordingChannelGroup" in name:
                        block.channel_indexes.append(self._read_recordingchannelgroup(child_node, parent=block))
                self._resolve_channel_indexes(block)
            elif self.merge_singles:
                # if no RecordingChannelGroups are defined, merging
                # takes place here.
                for segment in block.segments:
                    if hasattr(segment, 'unmerged_analogsignals'):
                        segment.analogsignals.extend(
                                self._merge_data_objects(segment.unmerged_analogsignals))
                        del segment.unmerged_analogsignals
                    if hasattr(segment, 'unmerged_irregularlysampledsignals'):
                        segment.irregularlysampledsignals.extend(
                                self._merge_data_objects(segment.unmerged_irregularlysampledsignals))
                        del segment.unmerged_irregularlysampledsignals
        return block

    def _read_segment(self, node, parent):
        attributes = self._get_standard_attributes(node)
        segment = Segment(**attributes)

        signals = []
        for name, child_node in node['analogsignals'].items():
            if "AnalogSignal" in name:
                signals.append(self._read_analogsignal(child_node, parent=segment))
        if signals and self.merge_singles:
            segment.unmerged_analogsignals = signals  # signals will be merged later
            signals = []
        for name, child_node in node['analogsignalarrays'].items():
            if "AnalogSignalArray" in name:
                signals.append(self._read_analogsignalarray(child_node, parent=segment))
        segment.analogsignals = signals

        irr_signals = []
        for name, child_node in node['irregularlysampledsignals'].items():
            if "IrregularlySampledSignal" in name:
                irr_signals.append(self._read_irregularlysampledsignal(child_node, parent=segment))
        if irr_signals and self.merge_singles:
            segment.unmerged_irregularlysampledsignals = irr_signals
            irr_signals = []
        segment.irregularlysampledsignals = irr_signals

        epochs = []
        for name, child_node in node['epochs'].items():
            if "Epoch" in name:
                epochs.append(self._read_epoch(child_node, parent=segment))
        if self.merge_singles:
            epochs = self._merge_data_objects(epochs)
        for name, child_node in node['epocharrays'].items():
            if "EpochArray" in name:
                epochs.append(self._read_epocharray(child_node, parent=segment))
        segment.epochs = epochs

        events = []
        for name, child_node in node['events'].items():
            if "Event" in name:
                events.append(self._read_event(child_node, parent=segment))
        if self.merge_singles:
            events = self._merge_data_objects(events)
        for name, child_node in node['eventarrays'].items():
            if "EventArray" in name:
                events.append(self._read_eventarray(child_node, parent=segment))
        segment.events = events

        spiketrains = []
        for name, child_node in node['spikes'].items():
            raise NotImplementedError('Spike objects not yet handled.')
        for name, child_node in node['spiketrains'].items():
            if "SpikeTrain" in name:
                spiketrains.append(self._read_spiketrain(child_node, parent=segment))
        segment.spiketrains = spiketrains

        segment.block = parent
        return segment

    def _read_analogsignalarray(self, node, parent):
        attributes = self._get_standard_attributes(node)
        # todo: handle channel_index
        sampling_rate = self._get_quantity(node["sampling_rate"])
        t_start = self._get_quantity(node["t_start"])
        signal = AnalogSignal(self._get_quantity(node["signal"]),
                              sampling_rate=sampling_rate, t_start=t_start,
                              **attributes)
        if self._lazy:
            signal.lazy_shape = node["signal"].shape
            if len(signal.lazy_shape) == 1:
                signal.lazy_shape = (signal.lazy_shape[0], 1)
        signal.segment = parent
        self.object_refs[node.attrs["object_ref"]] = signal
        return signal

    def _read_analogsignal(self, node, parent):
        return self._read_analogsignalarray(node, parent)

    def _read_irregularlysampledsignal(self, node, parent):
        attributes = self._get_standard_attributes(node)
        signal = IrregularlySampledSignal(times=self._get_quantity(node["times"]),
                                          signal=self._get_quantity(node["signal"]),
                                          **attributes)
        signal.segment = parent
        if self._lazy:
            signal.lazy_shape = node["signal"].shape
            if len(signal.lazy_shape) == 1:
                signal.lazy_shape = (signal.lazy_shape[0], 1)
        return signal

    def _read_spiketrain(self, node, parent):
        attributes = self._get_standard_attributes(node)
        t_start = self._get_quantity(node["t_start"])
        t_stop = self._get_quantity(node["t_stop"])
        # todo: handle sampling_rate, waveforms, left_sweep
        spiketrain = SpikeTrain(self._get_quantity(node["times"]),
                                t_start=t_start, t_stop=t_stop,
                                **attributes)
        spiketrain.segment = parent
        if self._lazy:
            spiketrain.lazy_shape = node["times"].shape
        self.object_refs[node.attrs["object_ref"]] = spiketrain
        return spiketrain

    def _read_epocharray(self, node, parent):
        attributes = self._get_standard_attributes(node)
        times = self._get_quantity(node["times"])
        durations = self._get_quantity(node["durations"])
        if self._lazy:
            labels = np.array((), dtype=node["labels"].dtype)
        else:
            labels = node["labels"].value
        epoch = Epoch(times=times, durations=durations, labels=labels, **attributes)
        epoch.segment = parent
        if self._lazy:
            epoch.lazy_shape = node["times"].shape
        return epoch

    def _read_epoch(self, node, parent):
        return self._read_epocharray(node, parent)

    def _read_eventarray(self, node, parent):
        attributes = self._get_standard_attributes(node)
        times = self._get_quantity(node["times"])
        if self._lazy:
            labels = np.array((), dtype=node["labels"].dtype)
        else:
            labels = node["labels"].value
        event = Event(times=times, labels=labels, **attributes)
        event.segment = parent
        if self._lazy:
            event.lazy_shape = node["times"].shape
        return event

    def _read_event(self, node, parent):
        return self._read_eventarray(node, parent)

    def _read_recordingchannelgroup(self, node, parent):
        # todo: handle Units
        attributes = self._get_standard_attributes(node)
        channel_indexes = node["channel_indexes"].value
        channel_names = node["channel_names"].value

        if channel_indexes.size:
            if len(node['recordingchannels']) :
                raise MergeError("Cannot handle a RecordingChannelGroup which both has a "
                                 "'channel_indexes' attribute and contains "
                                 "RecordingChannel objects")
            raise NotImplementedError("todo")  # need to handle node['analogsignalarrays']
        else:
            channels = []
            for name, child_node in node['recordingchannels'].items():
                if "RecordingChannel" in name:
                    channels.append(self._read_recordingchannel(child_node))
            channel_index = ChannelIndex(None, **attributes)
            channel_index._channels = channels
            # construction of the index is deferred until we have processed
            # all RecordingChannelGroup nodes
            units = []
            for name, child_node in node['units'].items():
                if "Unit" in name:
                    units.append(self._read_unit(child_node, parent=channel_index))
            channel_index.units = units
        channel_index.block = parent
        return channel_index

    def _read_recordingchannel(self, node):
        attributes = self._get_standard_attributes(node)
        analogsignals = []
        irregsignals = []

        for name, child_node in node["analogsignals"].items():
            if "AnalogSignal" in name:
                obj_ref = child_node.attrs["object_ref"]
                analogsignals.append(obj_ref)
        for name, child_node in node["irregularlysampledsignals"].items():
            if "IrregularlySampledSignal" in name:
                obj_ref = child_node.attrs["object_ref"]
                irregsignals.append(obj_ref)
        return attributes['index'], analogsignals, irregsignals

    def _read_unit(self, node, parent):
        attributes = self._get_standard_attributes(node)
        spiketrains = []
        for name, child_node in node["spiketrains"].items():
            if "SpikeTrain" in name:
                obj_ref = child_node.attrs["object_ref"]
                spiketrains.append(self.object_refs[obj_ref])
        unit = Unit(**attributes)
        unit.channel_index = parent
        unit.spiketrains = spiketrains
        return unit

    def _merge_data_objects(self, objects):
        if len(objects) > 1:
            merged_objects = [objects.pop(0)]
            while objects:
                obj = objects.pop(0)
                try:
                    combined_obj_ref = merged_objects[-1].annotations['object_ref']
                    merged_objects[-1] = merged_objects[-1].merge(obj)
                    merged_objects[-1].annotations['object_ref'] = combined_obj_ref + "-" + obj.annotations['object_ref']
                except MergeError:
                    merged_objects.append(obj)
            for obj in merged_objects:
                self.object_refs[obj.annotations['object_ref']] = obj
            return merged_objects
        else:
            return objects

    def _get_quantity(self, node):
        if self._lazy and len(node.shape) > 0:
            value = np.array((), dtype=node.dtype)
        else:
            value = node.value
        unit_str = [x for x in node.attrs.keys() if "unit" in x][0].split("__")[1]
        units = getattr(pq, unit_str)
        return value * units

    def _get_standard_attributes(self, node):
        """Retrieve attributes"""
        attributes = {}
        for name in ('name', 'description', 'index', 'file_origin', 'object_ref'):
            if name in node.attrs:
                attributes[name] = node.attrs[name]
        for name in ('rec_datetime', 'file_datetime'):
            if name in node.attrs:
                attributes[name] = pickle.loads(node.attrs[name])
        attributes.update(pickle.loads(node.attrs['annotations']))
        return attributes

    def _resolve_channel_indexes(self, block):

        def disjoint_channel_indexes(channel_indexes):
            channel_indexes = channel_indexes[:]
            for ci1 in channel_indexes:
                signal_group1 = set(tuple(x[1]) for x in ci1._channels)  # this works only on analogsignals
                for ci2 in channel_indexes:                              # need to take irregularly sampled signals
                    signal_group2 = set(tuple(x[1]) for x in ci2._channels)  # into account too
                    if signal_group1 != signal_group2:
                        if signal_group2.issubset(signal_group1):
                            channel_indexes.remove(ci2)
                        elif signal_group1.issubset(signal_group2):
                            channel_indexes.remove(ci1)
            return channel_indexes

        principal_indexes = disjoint_channel_indexes(block.channel_indexes)
        for ci in principal_indexes:
            ids = []
            by_segment = {}
            for (index, analogsignals, irregsignals) in ci._channels:
                ids.append(index)  # note that what was called "index" in Neo 0.3/0.4 is "id" in Neo 0.5
                for signal_ref in analogsignals:
                    signal = self.object_refs[signal_ref]
                    segment_id = id(signal.segment)
                    if segment_id in by_segment:
                        by_segment[segment_id]['analogsignals'].append(signal)
                    else:
                        by_segment[segment_id] = {'analogsignals': [signal], 'irregsignals': []}
                for signal_ref in irregsignals:
                    signal = self.object_refs[signal_ref]
                    segment_id = id(signal.segment)
                    if segment_id in by_segment:
                        by_segment[segment_id]['irregsignals'].append(signal)
                    else:
                        by_segment[segment_id] = {'analogsignals': [], 'irregsignals': [signal]}

            assert len(ids) > 0
            if self.merge_singles:
                ci.channel_ids = np.array(ids)
                ci.index = np.arange(len(ids))
                for seg_id, segment_data in by_segment.items():

                    # get the segment object
                    segment = None
                    for seg in ci.block.segments:
                        if id(seg) == seg_id:
                            segment = seg
                            break
                    assert segment is not None

                    if segment_data['analogsignals']:
                        merged_signals = self._merge_data_objects(segment_data['analogsignals'])
                        assert len(merged_signals) == 1
                        merged_signals[0].channel_index = ci
                        merged_signals[0].annotations['object_ref'] = "-".join(obj.annotations['object_ref']
                                                                               for obj in segment_data['analogsignals'])
                        segment.analogsignals.extend(merged_signals)
                        ci.analogsignals = merged_signals

                    if segment_data['irregsignals']:
                        merged_signals = self._merge_data_objects(segment_data['irregsignals'])
                        assert len(merged_signals) == 1
                        merged_signals[0].channel_index = ci
                        merged_signals[0].annotations['object_ref'] = "-".join(obj.annotations['object_ref']
                                                                               for obj in segment_data['irregsignals'])
                        segment.irregularlysampledsignals.extend(merged_signals)
                        ci.irregularlysampledsignals = merged_signals
            else:
                raise NotImplementedError()  # will need to return multiple ChannelIndexes

        # handle non-principal channel indexes
        for ci in block.channel_indexes:
            if ci not in principal_indexes:
                ids = [c[0] for c in ci._channels]
                for cipr in principal_indexes:
                    if ids[0] in cipr.channel_ids:
                        break
                ci.analogsignals = cipr.analogsignals
                ci.channel_ids = np.array(ids)
                ci.index = np.where(np.in1d(cipr.channel_ids, ci.channel_ids))[0]