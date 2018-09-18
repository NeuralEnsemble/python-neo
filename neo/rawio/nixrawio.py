"""
RawIO Class for NIX files

The RawIO assumes all segments and all blocks have the same structure.
It supports all kinds of NEO objects.

Author: Chek Yin Choi
"""

from __future__ import print_function, division, absolute_import
from neo.rawio.baserawio import (BaseRawIO, _signal_channel_dtype,
                                 _unit_channel_dtype, _event_channel_dtype)
import numpy as np
try:
    import nixio as nix

    HAVE_NIX = True
except ImportError:
    HAVE_NIX = False
    nix = None


class NIXRawIO(BaseRawIO):

    extensions = ['nix']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):

        self.file = nix.File.open(self.filename, nix.FileMode.ReadOnly)
        sig_channels = []
        size_list = []
        for bl in self.file.blocks:
            for seg in bl.groups:
                for da_idx, da in enumerate(seg.data_arrays):
                    if da.type == "neo.analogsignal":
                        chan_id = da_idx
                        ch_name = da.metadata['neo_name']
                        units = str(da.unit)
                        dtype = str(da.dtype)
                        sr = 1 / da.dimensions[0].sampling_interval
                        da_leng = da.size
                        if da_leng not in size_list:
                            size_list.append(da_leng)
                        group_id = 0
                        for sid, li_leng in enumerate(size_list):
                            if li_leng == da_leng:
                                group_id = sid
                                # very important! group_id use to store channel groups!!!
                                # use only for different signal length
                        gain = 1
                        offset = 0.
                        sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, group_id))
                break
            break
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        unit_channels = []
        unit_name = ""
        unit_id = ""
        for bl in self.file.blocks:
            for seg in bl.groups:
                for mt in seg.multi_tags:
                    if mt.type == "neo.spiketrain":
                        unit_name = mt.metadata['neo_name']
                        unit_id = mt.id
                        if mt.features:
                            wf_units = mt.features[0].data.unit
                            wf_sampling_rate = 1 / mt.features[0].data.dimensions[
                                2].sampling_interval
                        else:
                            wf_units = None
                            wf_sampling_rate = 0
                        wf_gain = 1
                        wf_offset = 0.
                        if mt.features and "left_sweep" in mt.features[0].data.metadata:
                            wf_left_sweep = mt.features[0].data.metadata["left_sweep"]
                        else:
                            wf_left_sweep = 0
                        unit_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                              wf_offset, wf_left_sweep, wf_sampling_rate))
                break
            break
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)

        event_channels = []
        event_count = 0
        epoch_count = 0
        for bl in self.file.blocks:
            for seg in bl.groups:
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
            d = {'segments': []}
            self.da_list['blocks'].append(d)
            for seg_index, seg in enumerate(blk.groups):
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
                self.da_list['blocks'][block_index]['segments'][seg_index]['data_size'] = size_list
                self.da_list['blocks'][block_index]['segments'][seg_index]['data'] = data_list
                self.da_list['blocks'][block_index]['segments'][seg_index]['ch_name'] = da_name_list

        self.unit_list = {'blocks': []}
        for block_index, blk in enumerate(self.file.blocks):
            d = {'segments': []}
            self.unit_list['blocks'].append(d)
            for seg_index, seg in enumerate(blk.groups):
                d = {'spiketrains': [], 'spiketrains_id': [], 'spiketrains_unit': []}
                self.unit_list['blocks'][block_index]['segments'].append(d)
                st_idx = 0
                for st in seg.multi_tags:
                    d = {'waveforms': []}
                    self.unit_list['blocks'][block_index]['segments'][seg_index]\
                        ['spiketrains_unit'].append(d)
                    if st.type == 'neo.spiketrain':
                        seg = self.unit_list['blocks'][block_index]['segments'][seg_index]
                        seg['spiketrains'].append(st.positions)
                        seg['spiketrains_id'].append(st.id)
                        if st.features and st.features[0].data.type == "neo.waveforms":
                            waveforms = st.features[0].data
                            if waveforms:
                                seg['spiketrains_unit'][st_idx]['waveforms'] = waveforms
                            else:
                                seg['spiketrains_unit'][st_idx]['waveforms'] = None
                            # assume one spiketrain one waveform
                            st_idx += 1

        self.header = {}
        self.header['nb_block'] = len(self.file.blocks)
        self.header['nb_segment'] = [len(bl.groups) for bl in self.file.blocks]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

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

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        if channel_indexes is None:
            channel_indexes = list(range(self.header['signal_channels'].size))
        ch_idx = channel_indexes[0]
        size = self.da_list['blocks'][block_index]['segments'][seg_index]['data_size'][ch_idx]
        return size  # size is per signal, not the sum of all channel_indexes

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        if channel_indexes is None:
            channel_indexes = list(range(self.header['signal_channels'].size))
        ch_idx = channel_indexes[0]
        da = [da for da in self.file.blocks[block_index].groups[seg_index].data_arrays][ch_idx]
        sig_t_start = float(da.metadata['t_start'])
        return sig_t_start  # assume same group_id always same t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):

        if channel_indexes is None:
            channel_indexes = list(range(self.header['signal_channels'].size))
        if i_start is None:
            i_start = 0
        if i_stop is None:
            for c in channel_indexes:
                i_stop = self.da_list['blocks'][block_index]['segments'][seg_index]['data_size'][c]
                break

        raw_signals_list = []
        da_list = self.da_list['blocks'][block_index]['segments'][seg_index]
        for idx in channel_indexes:
            da = da_list['data'][idx]
            raw_signals_list.append(da[i_start:i_stop])

        raw_signals = np.array(raw_signals_list)
        raw_signals = np.transpose(raw_signals)
        return raw_signals

    def _spike_count(self, block_index, seg_index, unit_index):
        count = 0
        head_id = self.header['unit_channels'][unit_index][1]
        for mt in self.file.blocks[block_index].groups[seg_index].multi_tags:
            for src in mt.sources:
                if mt.type == 'neo.spiketrain' and [src.type == "neo.unit"]:
                    if head_id == src.id:
                        return len(mt.positions)
        return count

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        spike_dict = self.unit_list['blocks'][block_index]['segments'][seg_index]['spiketrains']
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

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        # this must return a 3D numpy array (nb_spike, nb_channel, nb_sample)
        seg = self.unit_list['blocks'][block_index]['segments'][seg_index]
        waveforms = seg['spiketrains_unit'][unit_index]['waveforms']
        if not waveforms:
            return None
        raw_waveforms = np.array(waveforms)

        if t_start is not None:
            lim0 = t_start
            mask = (raw_waveforms >= lim0)
            raw_waveforms = np.where(mask, raw_waveforms, np.nan)  # use nan to keep the shape
        if t_stop is not None:
            lim1 = t_stop
            mask = (raw_waveforms <= lim1)
            raw_waveforms = np.where(mask, raw_waveforms, np.nan)
        return raw_waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        event_count = 0
        for event in self.file.blocks[block_index].groups[seg_index].multi_tags:
            if event.type == 'neo.event' or event.type == 'neo.epoch':
                if event_count == event_channel_index:
                    return len(event.positions)
                else:
                    event_count += 1
        return event_count

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        timestamp = []
        labels = []
        durations = None
        if event_channel_index == None:
            raise IndexError
        for mt in self.file.blocks[block_index].groups[seg_index].multi_tags:
            if mt.type == "neo.event" or mt.type == "neo.epoch":
                labels.append(mt.positions.dimensions[0].labels)
                po = mt.positions
                if po.type == "neo.event.times" or po.type == "neo.epoch.times":
                    timestamp.append(po)
                if self.header['event_channels'][event_channel_index]\
                            ['type'] == b'epoch' and mt.extents:
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

    def _rescale_event_timestamp(self, event_timestamps, dtype='float64'):
        ev_unit = ''
        for mt in self.file.blocks[0].groups[0].multi_tags:
            if mt.type == "neo.event":
                ev_unit = mt.positions.unit
                break
        if ev_unit == 'ms':
            event_timestamps /= 1000
        event_times = event_timestamps.astype(dtype)
        # supposing unit is second, other possibilies maybe mS microS...
        return event_times  # return in seconds

    def _rescale_epoch_duration(self, raw_duration, dtype='float64'):
        ep_unit = ''
        for mt in self.file.blocks[0].groups[0].multi_tags:
            if mt.type == "neo.epoch":
                ep_unit = mt.positions.unit
                break
        if ep_unit == 'ms':
            raw_duration /= 1000
        durations = raw_duration.astype(dtype)
        # supposing unit is second, other possibilies maybe mS microS...
        return durations  # return in seconds
