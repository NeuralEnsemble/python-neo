# -*- coding: utf-8 -*-
"""
Classe for reading data in CED spike2 files (.smr).

This code is based on:
 - sonpy, written by Antonio Gonzalez <Antonio.Gonzalez@cantab.net>
    Disponible here ::
    http://www.neuro.ki.se/broberger/

and sonpy come from :
 - SON Library 2.0 for MATLAB, written by Malcolm Lidierth at
    King's College London.
    See http://www.kcl.ac.uk/depsta/biomedical/cfnr/lidierth.html

This IO support old (<v6) and new files (>v7) of spike2


Author: Samuel Garcia

"""
from __future__ import print_function, division, absolute_import
# from __future__ import unicode_literals is not compatible with numpy.dtype both py2 py3

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np
from collections import OrderedDict


class Spike2RawIO(BaseRawIO):
    """

    """
    extensions = ['smr']
    rawmode = 'one-file'

    def __init__(self, filename='', take_ideal_sampling_rate=False, ced_units=True):
        BaseRawIO.__init__(self)
        self.filename = filename

        self.take_ideal_sampling_rate = take_ideal_sampling_rate
        self.ced_units = ced_units

    def _parse_header(self):

        # get header info and channel_info
        with open(self.filename, 'rb') as fid:
            self._global_info = read_as_dict(fid, headerDescription)
            info = self._global_info
            if info['system_id'] < 6:
                info['dtime_base'] = 1e-6
                info['datetime_detail'] = 0
                info['datetime_year'] = 0

            self._time_factor = info['us_per_time'] * info['dtime_base']

            self._channel_infos = []
            for chan_id in range(info['channels']):
                fid.seek(512 + 140 * chan_id)
                chan_info = read_as_dict(fid, channelHeaderDesciption1)

                if chan_info['kind'] in [1, 6]:
                    dt = [('scale', 'f4'), ('offset', 'f4'), ('unit', 'S6'), ]
                    chan_info.update(read_as_dict(fid, dt))

                elif chan_info['kind'] in [7, 9]:
                    dt = [('min', 'f4'), ('max', 'f4'), ('unit', 'S6'), ]
                    chan_info.update(read_as_dict(fid, dt))

                elif chan_info['kind'] in [4]:
                    dt = [('init_low', 'u1'), ('next_low', 'u1'), ]
                    chan_info.update(read_as_dict(fid, dt))

                if chan_info['kind'] in [1, 6, 7, 9]:
                    if info['system_id'] < 6:
                        chan_info.update(read_as_dict(fid, [('divide', 'i2')]))
                    else:
                        chan_info.update(read_as_dict(fid, [('interleave', 'i2')]))

                chan_info['type'] = dict_kind[chan_info['kind']]

                if chan_info['blocks'] == 0:
                    chan_info['t_start'] = 0.  # this means empty signals
                else:
                    fid.seek(chan_info['firstblock'])
                    block_info = read_as_dict(fid, blockHeaderDesciption)
                    chan_info['t_start'] = block_info['start_time'] * \
                        info['us_per_time'] * info['dtime_base']

                self._channel_infos.append(chan_info)

        # get data blocks index for all channel
        # run through all data block of of channel to prepare chan to block maps
        self._memmap = np.memmap(self.filename, dtype='u1', offset=0, mode='r')
        self._all_data_blocks = {}
        self._by_seg_data_blocks = {}
        for chan_id, chan_info in enumerate(self._channel_infos):
            data_blocks = []
            ind = chan_info['firstblock']
            for b in range(chan_info['blocks']):
                block_info = self._memmap[ind:ind + 20].view(blockHeaderDesciption)[0]
                data_blocks.append((ind, block_info['items'], 0,
                                    block_info['start_time'], block_info['end_time']))
                ind = block_info['succ_block']

            data_blocks = np.array(data_blocks, dtype=[(
                'pos', 'int32'), ('size', 'int32'), ('cumsum', 'int32'),
                ('start_time', 'int32'), ('end_time', 'int32')])
            data_blocks['pos'] += 20  # 20 is ths header size

            self._all_data_blocks[chan_id] = data_blocks
            self._by_seg_data_blocks[chan_id] = []

        # For all signal channel detect gaps between data block (pause in rec) so new Segment.
        # then check that all channel have the same gaps.
        # this part is tricky because we need to check that all channel have same pause.
        all_gaps_block_ind = {}
        for chan_id, chan_info in enumerate(self._channel_infos):
            if chan_info['kind'] in [1, 9]:
                data_blocks = self._all_data_blocks[chan_id]
                sig_size = np.sum(self._all_data_blocks[chan_id]['size'])
                if sig_size > 0:
                    interval = get_sample_interval(info, chan_info) / self._time_factor
                    # detect gaps
                    inter_block_sizes = data_blocks['start_time'][1:] - \
                        data_blocks['end_time'][:-1]
                    gaps_block_ind, = np.nonzero(inter_block_sizes > interval)
                    all_gaps_block_ind[chan_id] = gaps_block_ind

        # find t_start/t_stop for each seg based on gaps indexe
        self._sig_t_starts = {}
        self._sig_t_stops = {}
        if len(all_gaps_block_ind) == 0:
            # this means no signal channels
            nb_segment = 1
            # loop over event/spike channel to get the min/max time
            t_start, t_stop = None, None
            for chan_id, chan_info in enumerate(self._channel_infos):
                data_blocks = self._all_data_blocks[chan_id]
                if data_blocks.size > 0:
                    # if t_start is None or data_blocks[0]['start_time']<t_start:
                    # t_start = data_blocks[0]['start_time']
                    if t_stop is None or data_blocks[-1]['end_time'] > t_stop:
                        t_stop = data_blocks[-1]['end_time']
            # self._seg_t_starts = [t_start]
            self._seg_t_starts = [0]
            self._seg_t_stops = [t_stop]
        else:
            all_nb_seg = np.array([v.size + 1 for v in all_gaps_block_ind.values()])
            assert np.all(all_nb_seg[0] == all_nb_seg), \
                'Signal channel have differents pause so diffrents nb_segment'
            nb_segment = int(all_nb_seg[0])

            for chan_id, gaps_block_ind in all_gaps_block_ind.items():
                data_blocks = self._all_data_blocks[chan_id]
                self._sig_t_starts[chan_id] = []
                self._sig_t_stops[chan_id] = []

                for seg_ind in range(nb_segment):
                    if seg_ind == 0:
                        fisrt_bl = 0
                    else:
                        fisrt_bl = gaps_block_ind[seg_ind - 1] + 1
                    self._sig_t_starts[chan_id].append(data_blocks[fisrt_bl]['start_time'])

                    if seg_ind < nb_segment - 1:
                        last_bl = gaps_block_ind[seg_ind]
                    else:
                        last_bl = data_blocks.size - 1

                    self._sig_t_stops[chan_id].append(data_blocks[last_bl]['end_time'])

                    in_seg_data_block = data_blocks[fisrt_bl:last_bl + 1]
                    in_seg_data_block['cumsum'][1:] = np.cumsum(in_seg_data_block['size'][:-1])
                    self._by_seg_data_blocks[chan_id].append(in_seg_data_block)

            self._seg_t_starts = []
            self._seg_t_stops = []
            for seg_ind in range(nb_segment):
                # there is a small delay between all channel so take the max/min for t_start/t_stop
                t_start = min(
                    self._sig_t_starts[chan_id][seg_ind] for chan_id in self._sig_t_starts)
                t_stop = max(self._sig_t_stops[chan_id][seg_ind] for chan_id in self._sig_t_stops)
                self._seg_t_starts.append(t_start)
                self._seg_t_stops.append(t_stop)

        # create typed channels
        sig_channels = []
        unit_channels = []
        event_channels = []

        self.internal_unit_ids = {}
        for chan_id, chan_info in enumerate(self._channel_infos):
            if chan_info['kind'] in [1, 6, 7, 9]:
                if self.take_ideal_sampling_rate:
                    sampling_rate = info['ideal_rate']
                else:
                    sample_interval = get_sample_interval(info, chan_info)
                    sampling_rate = (1. / sample_interval)

            name = chan_info['title']

            if chan_info['kind'] in [1, 9]:
                # AnalogSignal
                if chan_id not in self._sig_t_starts:
                    continue
                units = chan_info['unit']
                if chan_info['kind'] == 1:  # int16
                    gain = chan_info['scale'] / 6553.6
                    offset = chan_info['offset']
                    sig_dtype = 'int16'
                elif chan_info['kind'] == 9:  # float32
                    gain = 1.
                    offset = 0.
                    sig_dtype = 'int32'
                group_id = 0
                sig_channels.append((name, chan_id, sampling_rate, sig_dtype,
                                     units, gain, offset, group_id))

            elif chan_info['kind'] in [2, 3, 4, 5, 8]:
                # Event
                event_channels.append((name, chan_id, 'event'))

            elif chan_info['kind'] in [6, 7]:  # SpikeTrain with waveforms
                wf_units = chan_info['unit']
                if chan_info['kind'] == 6:
                    wf_gain = chan_info['scale'] / 6553.6
                    wf_offset = chan_info['offset']
                    wf_left_sweep = chan_info['n_extra'] // 4
                elif chan_info['kind'] == 7:
                    wf_gain = 1.
                    wf_offset = 0.
                    wf_left_sweep = chan_info['n_extra'] // 8
                wf_sampling_rate = sampling_rate
                if self.ced_units:
                    # this is a hudge pain because need
                    # to jump over all blocks
                    data_blocks = self._all_data_blocks[chan_id]
                    dt = get_channel_dtype(chan_info)
                    unit_ids = set()
                    for bl in range(data_blocks.size):
                        ind0 = data_blocks[bl]['pos']
                        ind1 = data_blocks[bl]['size'] * dt.itemsize + ind0
                        raw_data = self._memmap[ind0:ind1].view(dt)
                        marker = raw_data['marker'] & 255
                        unit_ids.update(np.unique(marker))
                    unit_ids = sorted(list(unit_ids))
                else:
                    # All spike from one channel are group in one SpikeTrain
                    unit_ids = ['all']
                for unit_id in unit_ids:
                    unit_index = len(unit_channels)
                    self.internal_unit_ids[unit_index] = (chan_id, unit_id)
                    _id = "ch{}#{}".format(chan_id, unit_id)
                    unit_channels.append((name, _id, wf_units, wf_gain, wf_offset,
                                          wf_left_sweep, wf_sampling_rate))

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        if len(sig_channels) > 0:
            # signal channel can different sampling_rate/dtype/t_start/signal_length...
            # grouping them is difficults, so each channe = one group

            sig_channels['group_id'] = np.arange(sig_channels.size)
            self._sig_dtypes = {s['group_id']: np.dtype(s['dtype']) for s in sig_channels}

        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [nb_segment]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        # Annotations
        self._generate_minimal_annotations()
        bl_ann = self.raw_annotations['blocks'][0]
        bl_ann['system_id'] = info['system_id']
        seg_ann = bl_ann['segments'][0]
        seg_ann['system_id'] = info['system_id']

        for c, sig_channel in enumerate(sig_channels):
            chan_id = sig_channel['id']
            anasig_an = seg_ann['signals'][c]
            anasig_an['physical_channel_index'] = self._channel_infos[chan_id]['phy_chan']
            anasig_an['comment'] = self._channel_infos[chan_id]['comment']

        for c, unit_channel in enumerate(unit_channels):
            chan_id, unit_id = self.internal_unit_ids[c]
            unit_an = seg_ann['units'][c]
            unit_an['physical_channel_index'] = self._channel_infos[chan_id]['phy_chan']
            unit_an['comment'] = self._channel_infos[chan_id]['comment']

        for c, event_channel in enumerate(event_channels):
            chan_id = int(event_channel['id'])
            ev_an = seg_ann['events'][c]
            ev_an['physical_channel_index'] = self._channel_infos[chan_id]['phy_chan']
            ev_an['comment'] = self._channel_infos[chan_id]['comment']

    def _source_name(self):
        return self.filename

    def _segment_t_start(self, block_index, seg_index):
        return self._seg_t_starts[seg_index] * self._time_factor

    def _segment_t_stop(self, block_index, seg_index):
        return self._seg_t_stops[seg_index] * self._time_factor

    def _check_channel_indexes(self, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        channel_indexes = np.arange(self.header['signal_channels'].size)[channel_indexes]
        assert len(channel_indexes) == 1
        return channel_indexes

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        channel_indexes = self._check_channel_indexes(channel_indexes)
        chan_id = self.header['signal_channels'][channel_indexes[0]]['id']
        sig_size = np.sum(self._by_seg_data_blocks[chan_id][seg_index]['size'])
        return sig_size

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        channel_indexes = self._check_channel_indexes(channel_indexes)
        chan_id = self.header['signal_channels'][channel_indexes[0]]['id']
        return self._sig_t_starts[chan_id][seg_index] * self._time_factor

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._get_signal_size(block_index, seg_index, channel_indexes)

        channel_indexes = self._check_channel_indexes(channel_indexes)
        chan_index = channel_indexes[0]
        chan_id = self.header['signal_channels'][chan_index]['id']
        group_id = self.header['signal_channels'][channel_indexes[0]]['group_id']
        dt = self._sig_dtypes[group_id]

        raw_signals = np.zeros((i_stop - i_start, len(channel_indexes)), dtype=dt)
        for c, channel_index in enumerate(channel_indexes):
            # NOTE: this actual way is slow because we run throught
            # the file for each channel. The loop should be reversed.
            # But there is no garanty that channels shared the same data block
            # indexes. So this make the job too difficult.
            chan_header = self.header['signal_channels'][channel_index]
            chan_id = chan_header['id']
            data_blocks = self._by_seg_data_blocks[chan_id][seg_index]

            # loop over data blocks and get chunks
            bl0 = np.searchsorted(data_blocks['cumsum'], i_start, side='left')
            bl1 = np.searchsorted(data_blocks['cumsum'], i_stop, side='left')
            ind = 0
            for bl in range(bl0, bl1):
                ind0 = data_blocks[bl]['pos']
                ind1 = data_blocks[bl]['size'] * dt.itemsize + ind0
                data = self._memmap[ind0:ind1].view(dt)
                if bl == bl1 - 1:
                    # right border
                    # be carfull that bl could be both bl0 and bl1!!
                    border = data.size - (i_stop - data_blocks[bl]['cumsum'])
                    if border > 0:
                        data = data[:-border]
                if bl == bl0:
                    # left border
                    border = i_start - data_blocks[bl]['cumsum']
                    data = data[border:]
                raw_signals[ind:data.size + ind, c] = data
                ind += data.size
        return raw_signals

    def _count_in_time_slice(self, seg_index, chan_id, lim0, lim1, marker_filter=None):
        # count event or spike in time slice
        data_blocks = self._all_data_blocks[chan_id]
        chan_info = self._channel_infos[chan_id]
        dt = get_channel_dtype(chan_info)
        nb = 0
        for bl in range(data_blocks.size):
            ind0 = data_blocks[bl]['pos']
            ind1 = data_blocks[bl]['size'] * dt.itemsize + ind0
            raw_data = self._memmap[ind0:ind1].view(dt)
            ts = raw_data['tick']
            keep = (ts >= lim0) & (ts <= lim1)
            if marker_filter is not None:
                keep2 = (raw_data['marker'] & 255) == marker_filter
                keep = keep & keep2
            nb += np.sum(keep)
            if ts[-1] > lim1:
                break
        return nb

    def _get_internal_timestamp_(self, seg_index, chan_id,
                                 t_start, t_stop, other_field=None, marker_filter=None):
        chan_info = self._channel_infos[chan_id]
        # data_blocks = self._by_seg_data_blocks[chan_id][seg_index]
        data_blocks = self._all_data_blocks[chan_id]
        dt = get_channel_dtype(chan_info)

        if t_start is None:
            # lim0 = 0
            lim0 = self._seg_t_starts[seg_index]
        else:
            lim0 = int(t_start / self._time_factor)

        if t_stop is None:
            # lim1 = 2**32
            lim1 = self._seg_t_stops[seg_index]
        else:
            lim1 = int(t_stop / self._time_factor)

        timestamps = []
        othervalues = []
        for bl in range(data_blocks.size):
            ind0 = data_blocks[bl]['pos']
            ind1 = data_blocks[bl]['size'] * dt.itemsize + ind0
            raw_data = self._memmap[ind0:ind1].view(dt)
            ts = raw_data['tick']
            keep = (ts >= lim0) & (ts <= lim1)
            if marker_filter is not None:
                keep2 = (raw_data['marker'] & 255) == marker_filter
                keep = keep & keep2

            timestamps.append(ts[keep])
            if other_field is not None:
                othervalues.append(raw_data[other_field][keep])
            if ts[-1] > lim1:
                break

        if len(timestamps) > 0:
            timestamps = np.concatenate(timestamps)
        else:
            timestamps = np.zeros(0, dtype='int16')

        if other_field is None:
            return timestamps
        else:
            if len(timestamps) > 0:
                othervalues = np.concatenate(othervalues)
            else:
                othervalues = np.zeros(0, dtype=dt.fields[other_field][0])
            return timestamps, othervalues

    def _spike_count(self, block_index, seg_index, unit_index):
        chan_id, unit_id = self.internal_unit_ids[unit_index]
        if self.ced_units:
            marker_filter = unit_id
        else:
            marker_filter = None
        lim0 = self._seg_t_starts[seg_index]
        lim1 = self._seg_t_stops[seg_index]
        return self._count_in_time_slice(seg_index, chan_id,
                                         lim0, lim1, marker_filter=marker_filter)

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        unit_header = self.header['unit_channels'][unit_index]
        chan_id, unit_id = self.internal_unit_ids[unit_index]

        if self.ced_units:
            marker_filter = unit_id
        else:
            marker_filter = None

        spike_timestamps = self._get_internal_timestamp_(seg_index,
                                                         chan_id, t_start, t_stop,
                                                         marker_filter=marker_filter)

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times *= self._time_factor
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        unit_header = self.header['unit_channels'][unit_index]
        chan_id, unit_id = self.internal_unit_ids[unit_index]

        if self.ced_units:
            marker_filter = unit_id
        else:
            marker_filter = None

        timestamps, waveforms = self._get_internal_timestamp_(seg_index, chan_id,
                                                              t_start, t_stop,
                                                              other_field='waveform',
                                                              marker_filter=marker_filter)

        waveforms = waveforms.reshape(timestamps.size, 1, -1)

        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        event_header = self.header['event_channels'][event_channel_index]
        chan_id = int(event_header['id'])  # because set to string in header
        lim0 = self._seg_t_starts[seg_index]
        lim1 = self._seg_t_stops[seg_index]
        return self._count_in_time_slice(seg_index, chan_id, lim0, lim1, marker_filter=None)

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        event_header = self.header['event_channels'][event_channel_index]
        chan_id = int(event_header['id'])  # because set to string in header
        chan_info = self._channel_infos[chan_id]

        if chan_info['kind'] == 5:
            timestamps, labels = self._get_internal_timestamp_(seg_index,
                                                               chan_id, t_start, t_stop,
                                                               other_field='marker')
        elif chan_info['kind'] == 8:
            timestamps, labels = self._get_internal_timestamp_(seg_index,
                                                               chan_id, t_start, t_stop,
                                                               other_field='label')
        else:
            timestamps = self._get_internal_timestamp_(seg_index,
                                                       chan_id, t_start, t_stop, other_field=None)
            labels = np.zeros(timestamps.size, dtype='U')

        labels = labels.astype('U')
        durations = None

        return timestamps, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        event_times = event_timestamps.astype(dtype)
        event_times *= self._time_factor
        return event_times


def read_as_dict(fid, dtype):
    """
    Given a file descriptor (seek at the good place externally)
    and a numpy.dtype of the binary struct return a dict.
    Make conversion for strings.
    """
    dt = np.dtype(dtype)
    h = np.fromstring(fid.read(dt.itemsize), dt)[0]
    info = OrderedDict()
    for k in dt.names:
        v = h[k]

        if dt[k].kind == 'S':
            v = v.decode('iso-8859-1')
            if len(v) > 0:
                l = ord(v[0])
                v = v[1:l + 1]

        info[k] = v
    return info


def get_channel_dtype(chan_info):
    """
    Get dtype by kind.
    """
    if chan_info['kind'] == 1:  # Raw signal
        dt = 'int16'
    elif chan_info['kind'] in [2, 3, 4]:  # Event data
        dt = [('tick', 'i4')]
    elif chan_info['kind'] in [5]:  # Marker data
        dt = [('tick', 'i4'), ('marker', 'i4')]
    elif chan_info['kind'] in [6]:  # AdcMark data (waveform)
        dt = [('tick', 'i4'), ('marker', 'i4'),
              # ('adc', 'S%d' % chan_info['n_extra'])]
              ('waveform', 'int16', chan_info['n_extra'] // 2)]
    elif chan_info['kind'] in [7]:  # RealMark data (waveform)
        dt = [('tick', 'i4'), ('marker', 'i4'),
              # ('real', 'S%d' % chan_info['n_extra'])]
              ('waveform', 'float32', chan_info['n_extra'] // 4)]
    elif chan_info['kind'] in [8]:  # TextMark data
        dt = [('tick', 'i4'), ('marker', 'i4'),
              ('label', 'S%d' % chan_info['n_extra'])]
    elif chan_info['kind'] == 9:  # Float signal
        dt = 'float32'
    dt = np.dtype(dt)
    return dt


def get_sample_interval(info, chan_info):
    """
    Get sample interval for one channel
    """
    if info['system_id'] in [1, 2, 3, 4, 5]:  # Before version 5
        sample_interval = (chan_info['divide'] * info['us_per_time'] *
                           info['time_per_adc']) * 1e-6
    else:
        sample_interval = (chan_info['l_chan_dvd'] *
                           info['us_per_time'] * info['dtime_base'])
    return sample_interval


# headers structures :
headerDescription = [
    ('system_id', 'i2'),
    ('copyright', 'S10'),
    ('creator', 'S8'),
    ('us_per_time', 'i2'),
    ('time_per_adc', 'i2'),
    ('filestate', 'i2'),
    ('first_data', 'i4'),  # i8
    ('channels', 'i2'),
    ('chan_size', 'i2'),
    ('extra_data', 'i2'),
    ('buffersize', 'i2'),
    ('os_format', 'i2'),
    ('max_ftime', 'i4'),  # i8
    ('dtime_base', 'f8'),
    ('datetime_detail', 'u1'),
    ('datetime_year', 'i2'),
    ('pad', 'S52'),
    ('comment1', 'S80'),
    ('comment2', 'S80'),
    ('comment3', 'S80'),
    ('comment4', 'S80'),
    ('comment5', 'S80'),
]

channelHeaderDesciption1 = [
    ('del_size', 'i2'),
    ('next_del_block', 'i4'),  # i8
    ('firstblock', 'i4'),  # i8
    ('lastblock', 'i4'),  # i8
    ('blocks', 'i2'),
    ('n_extra', 'i2'),
    ('pre_trig', 'i2'),
    ('free0', 'i2'),
    ('py_sz', 'i2'),
    ('max_data', 'i2'),
    ('comment', 'S72'),
    ('max_chan_time', 'i4'),  # i8
    ('l_chan_dvd', 'i4'),  # i8
    ('phy_chan', 'i2'),
    ('title', 'S10'),
    ('ideal_rate', 'f4'),
    ('kind', 'u1'),
    ('unused1', 'i1'),
]

blockHeaderDesciption = [
    ('pred_block', 'i4'),  # i8
    ('succ_block', 'i4'),  # i8
    ('start_time', 'i4'),  # i8
    ('end_time', 'i4'),  # i8
    ('channel_num', 'i2'),
    ('items', 'i2'),
]

dict_kind = {
    0: 'empty',
    1: 'Adc',
    2: 'EventFall',
    3: 'EventRise',
    4: 'EventBoth',
    5: 'Marker',
    6: 'AdcMark',
    7: 'RealMark',
    8: 'TextMark',
    9: 'RealWave',
}
