"""
RawIO Class for NIX files

The RawIO assumes all segments and all blocks have the same structure.
It supports all kinds of NEO objects.

Author: Chek Yin Choi, Julia Sprenger
"""

import os.path

import numpy as np

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)
from ..io.nixio import check_nix_version



# When reading metadata properties, the following keys are ignored since they
# are used to store Neo object properties.
# This dictionary is used in the _filter_properties() method.
neo_attributes = {
    "segment": ["index"],
    "analogsignal": ["units", "copy", "sampling_rate", "t_start"],
    "spiketrain": ["units", "copy", "sampling_rate", "t_start", "t_stop",
                   "waveforms", "left_sweep"],
    "event": ["times", "labels", "units", "durations", "copy"]
}


class NIXRawIO(BaseRawIO):

    extensions = ['nix', 'h5']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        check_nix_version()
        BaseRawIO.__init__(self)
        self.filename = str(filename)

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        import nixio

        self.file = nixio.File.open(self.filename, nixio.FileMode.ReadOnly)
        signal_channels = []
        anasig_ids = {0: []}  # ids of analogsignals by segment
        stream_ids = []
        for bl in self.file.blocks:
            for seg in bl.groups:
                for da_idx, da in enumerate(seg.data_arrays):
                    if da.type == "neo.analogsignal":
                        chan_id = da_idx
                        ch_name = da.metadata['neo_name']
                        units = str(da.unit)
                        dtype = str(da.dtype)
                        sr = 1 / da.dimensions[0].sampling_interval
                        anasig_id = da.name.split('.')[-2]
                        if anasig_id not in anasig_ids[0]:
                            anasig_ids[0].append(anasig_id)
                        stream_id = anasig_ids[0].index(anasig_id)
                        if stream_id not in stream_ids:
                            stream_ids.append(stream_id)
                        gain = 1
                        offset = 0.
                        signal_channels.append((ch_name, chan_id, sr, dtype,
                                            units, gain, offset, stream_id))
                # only read structure of first segment and assume the same
                # across segments
                break
            break
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        signal_streams = np.zeros(len(stream_ids), dtype=_signal_stream_dtype)
        signal_streams['id'] = stream_ids
        signal_streams['name'] = ''

        spike_channels = []
        unit_name = ""
        unit_id = ""
        for bl in self.file.blocks:
            seg_groups = [g for g in bl.groups if g.type == "neo.segment"]

            for seg in seg_groups:
                for mt in seg.multi_tags:
                    if mt.type == "neo.spiketrain":
                        unit_name = mt.metadata['neo_name']
                        unit_id = mt.id
                        wf_left_sweep = 0
                        wf_units = None
                        wf_sampling_rate = 0
                        if mt.features:
                            wf = mt.features[0].data
                            wf_units = wf.unit
                            dim = wf.dimensions[2]
                            interval = dim.sampling_interval
                            wf_sampling_rate = 1 / interval
                            if wf.metadata:
                                wf_left_sweep = wf.metadata["left_sweep"]
                        wf_gain = 1
                        wf_offset = 0.
                        spike_channels.append(
                            (unit_name, unit_id, wf_units, wf_gain,
                             wf_offset, wf_left_sweep, wf_sampling_rate)
                        )
                break
            break
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = []
        event_count = 0
        epoch_count = 0
        for bl in self.file.blocks:
            seg_groups = [g for g in bl.groups if g.type == "neo.segment"]
            for seg in seg_groups:
                for mt in seg.multi_tags:
                    if mt.type == "neo.event":
                        ev_name = mt.metadata['neo_name']
                        ev_id = event_count
                        event_count += 1
                        ev_type = "event"
                        event_channels.append((ev_name, ev_id, ev_type))
                    if mt.type == "neo.epoch":
                        ep_name = mt.metadata['neo_name']
                        ep_id = epoch_count
                        epoch_count += 1
                        ep_type = "epoch"
                        event_channels.append((ep_name, ep_id, ep_type))
                break
            break
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.da_list = {'blocks': []}
        for block_index, blk in enumerate(self.file.blocks):
            seg_groups = [g for g in blk.groups if g.type == "neo.segment"]
            d = {'segments': []}
            self.da_list['blocks'].append(d)
            for seg_index, seg in enumerate(seg_groups):
                d = {'signals': []}
                self.da_list['blocks'][block_index]['segments'].append(d)
                size_list = []
                data_list = []
                da_name_list = []
                for da in seg.data_arrays:
                    if da.type == 'neo.analogsignal':
                        size_list.append(da.size)
                        data_list.append(da)
                        da_name_list.append(da.metadata['neo_name'])
                block = self.da_list['blocks'][block_index]
                segment = block['segments'][seg_index]
                segment['data_size'] = size_list
                segment['data'] = data_list
                segment['ch_name'] = da_name_list

        self.unit_list = {'blocks': []}
        for block_index, blk in enumerate(self.file.blocks):
            seg_groups = [g for g in blk.groups if g.type == "neo.segment"]
            d = {'segments': []}
            self.unit_list['blocks'].append(d)
            for seg_index, seg in enumerate(seg_groups):
                d = {'spiketrains': [],
                     'spiketrains_id': [],
                     'spiketrains_unit': []}
                self.unit_list['blocks'][block_index]['segments'].append(d)
                st_idx = 0
                for st in seg.multi_tags:
                    d = {'waveforms': []}
                    block = self.unit_list['blocks'][block_index]
                    segment = block['segments'][seg_index]
                    segment['spiketrains_unit'].append(d)
                    if st.type == 'neo.spiketrain':
                        segment['spiketrains'].append(st.positions)
                        segment['spiketrains_id'].append(st.id)
                        wftypestr = "neo.waveforms"
                        if (st.features and st.features[0].data.type == wftypestr):
                            waveforms = st.features[0].data
                            stdict = segment['spiketrains_unit'][st_idx]
                            if waveforms:
                                stdict['waveforms'] = waveforms
                            else:
                                stdict['waveforms'] = None
                            # assume one spiketrain one waveform
                            st_idx += 1

        self.header = {}
        self.header['nb_block'] = len(self.file.blocks)
        self.header['nb_segment'] = [
            len(seg_groups)
            for bl in self.file.blocks
        ]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()
        for blk_idx, blk in enumerate(self.file.blocks):
            seg_groups = [g for g in blk.groups if g.type == "neo.segment"]
            bl_ann = self.raw_annotations['blocks'][blk_idx]
            props = blk.metadata.inherited_properties()
            bl_ann.update(self._filter_properties(props, "block"))
            for grp_idx, group in enumerate(seg_groups):
                seg_ann = bl_ann['segments'][grp_idx]
                props = group.metadata.inherited_properties()
                seg_ann.update(self._filter_properties(props, "segment"))

                sp_idx = 0
                ev_idx = 0
                for mt in group.multi_tags:
                    if mt.type == 'neo.spiketrain' and seg_ann['spikes']:
                        st_ann = seg_ann['spikes'][sp_idx]
                        props = mt.metadata.inherited_properties()
                        st_ann.update(self._filter_properties(props, 'spiketrain'))
                        sp_idx += 1
                    # if order is preserving, the annotations
                    # should go to the right place, need test
                    if mt.type == "neo.event" or mt.type == "neo.epoch":
                        if seg_ann['events'] != []:
                            event_ann = seg_ann['events'][ev_idx]
                            props = mt.metadata.inherited_properties()
                            event_ann.update(self._filter_properties(props, 'event'))
                            ev_idx += 1

                # adding array annotations to analogsignals
                annotated_anasigs = []
                sig_ann = seg_ann['signals']
                # this implementation relies on analogsignals always being
                # stored in the same stream order across segments
                stream_id = 0
                for da_idx, da in enumerate(group.data_arrays):
                    if da.type != "neo.analogsignal":
                        continue
                    anasig_id = da.name.split('.')[-2]
                    # skip already annotated signals as each channel already
                    # contains the complete set of annotations and
                    # array_annotations
                    if anasig_id in annotated_anasigs:
                        continue
                    annotated_anasigs.append(anasig_id)

                    # collect annotation properties
                    props = [p for p in da.metadata.props
                             if p.type != 'ARRAYANNOTATION']
                    props_dict = self._filter_properties(props, "analogsignal")
                    sig_ann[stream_id].update(props_dict)

                    # collect array annotation properties
                    props = [p for p in da.metadata.props
                             if p.type == 'ARRAYANNOTATION']
                    props_dict = self._filter_properties(props, "analogsignal")
                    sig_ann[stream_id]['__array_annotations__'].update(
                        props_dict)

                    stream_id += 1

    def _segment_t_start(self, block_index, seg_index):
        t_start = 0
        for mt in self.file.blocks[block_index].groups[seg_index].multi_tags:
            if mt.type == "neo.spiketrain":
                t_start = mt.metadata['t_start']
        return t_start

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = 0
        for mt in self.file.blocks[block_index].groups[seg_index].multi_tags:
            if mt.type == "neo.spiketrain":
                t_stop = mt.metadata['t_stop']
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self.header['signal_streams'][stream_index]['id']
        keep = self.header['signal_channels']['stream_id'] == stream_id
        channel_indexes, = np.nonzero(keep)
        ch_idx = channel_indexes[0]
        block = self.da_list['blocks'][block_index]
        segment = block['segments'][seg_index]
        size = segment['data_size'][ch_idx]
        return size  # size is per signal, not the sum of all channel_indexes

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        stream_id = self.header['signal_streams'][stream_index]['id']
        keep = self.header['signal_channels']['stream_id'] == stream_id
        channel_indexes, = np.nonzero(keep)
        ch_idx = channel_indexes[0]
        block = self.file.blocks[block_index]
        das = [da for da in block.groups[seg_index].data_arrays]
        da = das[ch_idx]
        sig_t_start = float(da.metadata['t_start'])
        return sig_t_start  # assume same group_id always same t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        stream_id = self.header['signal_streams'][stream_index]['id']
        keep = self.header['signal_channels']['stream_id'] == stream_id
        global_channel_indexes, = np.nonzero(keep)
        if channel_indexes is not None:
            global_channel_indexes = global_channel_indexes[channel_indexes]

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.get_signal_size(block_index, seg_index, stream_index)

        raw_signals_list = []
        da_list = self.da_list['blocks'][block_index]['segments'][seg_index]
        for idx in global_channel_indexes:
            da = da_list['data'][idx]
            raw_signals_list.append(da[i_start:i_stop])

        raw_signals = np.array(raw_signals_list)
        raw_signals = np.transpose(raw_signals)
        return raw_signals

    def _spike_count(self, block_index, seg_index, unit_index):
        count = 0
        head_id = self.header['spike_channels'][unit_index][1]
        for mt in self.file.blocks[block_index].groups[seg_index].multi_tags:
            for src in mt.sources:
                if mt.type == 'neo.spiketrain' and [src.type == "neo.unit"]:
                    if head_id == src.id:
                        return len(mt.positions)
        return count

    def _get_spike_timestamps(self, block_index, seg_index, unit_index,
                              t_start, t_stop):
        block = self.unit_list['blocks'][block_index]
        segment = block['segments'][seg_index]
        spike_dict = segment['spiketrains']
        spike_timestamps = spike_dict[unit_index]
        spike_timestamps = np.transpose(spike_timestamps)

        if t_start is not None or t_stop is not None:
            lim0 = t_start
            lim1 = t_stop
            mask = (spike_timestamps >= lim0) & (spike_timestamps <= lim1)
            spike_timestamps = spike_timestamps[mask]
        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index,
                                 t_start, t_stop):
        # this must return a 3D numpy array (nb_spike, nb_channel, nb_sample)
        seg = self.unit_list['blocks'][block_index]['segments'][seg_index]
        waveforms = seg['spiketrains_unit'][unit_index]['waveforms']
        if not waveforms:
            return None
        raw_waveforms = np.array(waveforms)

        if t_start is not None:
            lim0 = t_start
            mask = (raw_waveforms >= lim0)
            # use nan to keep the shape
            raw_waveforms = np.where(mask, raw_waveforms, np.nan)
        if t_stop is not None:
            lim1 = t_stop
            mask = (raw_waveforms <= lim1)
            raw_waveforms = np.where(mask, raw_waveforms, np.nan)
        return raw_waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        event_count = 0
        segment = self.file.blocks[block_index].groups[seg_index]
        for event in segment.multi_tags:
            if event.type == 'neo.event' or event.type == 'neo.epoch':
                if event_count == event_channel_index:
                    return len(event.positions)
                else:
                    event_count += 1
        return event_count

    def _get_event_timestamps(self, block_index, seg_index,
                              event_channel_index, t_start, t_stop):
        timestamp = []
        labels = []
        durations = None
        if event_channel_index is None:
            raise IndexError
        for mt in self.file.blocks[block_index].groups[seg_index].multi_tags:
            if mt.type == "neo.event" or mt.type == "neo.epoch":
                labels.append(mt.positions.dimensions[0].labels)
                po = mt.positions
                if (po.type == "neo.event.times" or po.type == "neo.epoch.times"):
                    timestamp.append(po)
                channel = self.header['event_channels'][event_channel_index]
                if channel['type'] == b'epoch' and mt.extents:
                    if mt.extents.type == 'neo.epoch.durations':
                        durations = np.array(mt.extents)
                        break
        timestamp = timestamp[event_channel_index][:]
        timestamp = np.array(timestamp, dtype="float")
        labels = labels[event_channel_index][:]
        labels = np.array(labels, dtype='U')
        if t_start is not None:
            keep = timestamp >= t_start
            timestamp, labels = timestamp[keep], labels[keep]

        if t_stop is not None:
            keep = timestamp <= t_stop
            timestamp, labels = timestamp[keep], labels[keep]
        return timestamp, durations, labels  # only the first fits in rescale

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        ev_unit = ''
        for mt in self.file.blocks[0].groups[0].multi_tags:
            if mt.type == "neo.event":
                ev_unit = mt.positions.unit
                break
        if ev_unit == 'ms':
            event_timestamps /= 1000
        event_times = event_timestamps.astype(dtype)
        # supposing unit is second, other possibilities maybe mS microS...
        return event_times  # return in seconds

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        ep_unit = ''
        for mt in self.file.blocks[0].groups[0].multi_tags:
            if mt.type == "neo.epoch":
                ep_unit = mt.positions.unit
                break
        if ep_unit == 'ms':
            raw_duration /= 1000
        durations = raw_duration.astype(dtype)
        # supposing unit is second, other possibilities maybe mS microS...
        return durations  # return in seconds

    def _filter_properties(self, properties, neo_type):
        """
        Takes a collection of NIX metadata properties and the name of a Neo
        type and returns a dictionary representing the Neo object annotations.
        Properties that represent the attributes of the Neo object type are
        filtered, based on the global 'neo_attributes' dictionary.
        """
        annotations = dict()
        attrs = neo_attributes.get(neo_type, list())
        for prop in properties:
            # filter neo_name explicitly
            if not (prop.name in attrs or prop.name == "neo_name"):
                values = prop.values
                if len(values) == 1:
                    values = values[0]
                annotations[str(prop.name)] = values
        return annotations
