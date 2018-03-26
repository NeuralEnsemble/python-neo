# -*- coding: utf-8 -*-
"""
Class for reading the old data format from Plexon
acquisition system (.plx)

Note that Plexon now use a new format PL2 which is NOT
supported by this IO.

Compatible with versions 100 to 106.
Other versions have not been tested.

This IO is developed thanks to the header file downloadable from:
http://www.plexon.com/software-downloads

This IO was rewritten in 2017 and this was a huge pain because
the underlying file format is really inefficient.
The rewrite is now based on numpy dtype and not on Python struct.
This should be faster.
If one day, somebody use it, consider to offer me a beer.


Author: Samuel Garcia

"""
from __future__ import print_function, division, absolute_import
# from __future__ import unicode_literals is not compatible with numpy.dtype both py2 py3

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np
from collections import OrderedDict
import datetime


class PlexonRawIO(BaseRawIO):
    extensions = ['plx']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):

        # global header
        with open(self.filename, 'rb') as fid:
            offset0 = 0
            global_header = read_as_dict(fid, GlobalHeader, offset=offset0)

        rec_datetime = datetime.datetime(global_header['Year'],
                                         global_header['Month'],
                                         global_header['Day'],
                                         global_header['Hour'],
                                         global_header['Minute'],
                                         global_header['Second'])

        # dsp channels header = spikes and waveforms
        nb_unit_chan = global_header['NumDSPChannels']
        offset1 = np.dtype(GlobalHeader).itemsize
        dspChannelHeaders = np.memmap(self.filename, dtype=DspChannelHeader, mode='r',
                                      offset=offset1, shape=(nb_unit_chan,))

        # event channel header
        nb_event_chan = global_header['NumEventChannels']
        offset2 = offset1 + np.dtype(DspChannelHeader).itemsize * nb_unit_chan
        eventHeaders = np.memmap(self.filename, dtype=EventChannelHeader, mode='r',
                                 offset=offset2, shape=(nb_event_chan,))

        # slow channel header = signal
        nb_sig_chan = global_header['NumSlowChannels']
        offset3 = offset2 + np.dtype(EventChannelHeader).itemsize * nb_event_chan
        slowChannelHeaders = np.memmap(self.filename, dtype=SlowChannelHeader, mode='r',
                                       offset=offset3, shape=(nb_sig_chan,))

        offset4 = offset3 + np.dtype(SlowChannelHeader).itemsize * nb_sig_chan

        # loop over data blocks and put them by type and channel
        block_headers = {1: {c: [] for c in dspChannelHeaders['Channel']},
                         4: {c: [] for c in eventHeaders['Channel']},
                         5: {c: [] for c in slowChannelHeaders['Channel']},
                         }
        block_pos = {1: {c: [] for c in dspChannelHeaders['Channel']},
                     4: {c: [] for c in eventHeaders['Channel']},
                     5: {c: [] for c in slowChannelHeaders['Channel']},
                     }
        data = self._memmap = np.memmap(self.filename, dtype='u1', offset=0, mode='r')
        pos = offset4
        while pos < data.size:
            bl_header = data[pos:pos + 16].view(DataBlockHeader)[0]
            length = bl_header['NumberOfWaveforms'] * bl_header['NumberOfWordsInWaveform'] * 2 + 16
            bl_type = int(bl_header['Type'])
            chan_id = int(bl_header['Channel'])
            block_headers[bl_type][chan_id].append(bl_header)
            block_pos[bl_type][chan_id].append(pos)
            pos += length

        self._last_timestamps = bl_header['UpperByteOf5ByteTimestamp'] * \
                                2 ** 32 + bl_header['TimeStamp']

        # ... and finalize them in self._data_blocks
        # for a faster acces depending on type (1, 4, 5)
        self._data_blocks = {}
        dt_base = [('pos', 'int64'), ('timestamp', 'int64'), ('size', 'int64')]
        dtype_by_bltype = {
            # Spikes and waveforms
            1: np.dtype(dt_base + [('unit_id', 'uint16'), ('n1', 'uint16'), ('n2', 'uint16'), ]),
            # Events
            4: np.dtype(dt_base + [('label', 'uint16'), ]),
            # Signals
            5: np.dtype(dt_base + [('cumsum', 'int64'), ]),
        }
        for bl_type in block_headers:
            self._data_blocks[bl_type] = {}
            for chan_id in block_headers[bl_type]:
                bl_header = np.array(block_headers[bl_type][chan_id], dtype=DataBlockHeader)
                bl_pos = np.array(block_pos[bl_type][chan_id], dtype='int64')

                timestamps = bl_header['UpperByteOf5ByteTimestamp'] * \
                             2 ** 32 + bl_header['TimeStamp']

                n1 = bl_header['NumberOfWaveforms']
                n2 = bl_header['NumberOfWordsInWaveform']
                dt = dtype_by_bltype[bl_type]
                data_block = np.empty(bl_pos.size, dtype=dt)
                data_block['pos'] = bl_pos + 16
                data_block['timestamp'] = timestamps
                data_block['size'] = n1 * n2 * 2

                if bl_type == 1:  # Spikes and waveforms
                    data_block['unit_id'] = bl_header['Unit']
                    data_block['n1'] = n1
                    data_block['n2'] = n2
                elif bl_type == 4:  # Events
                    data_block['label'] = bl_header['Unit']
                elif bl_type == 5:  # Signals
                    if data_block.size > 0:
                        # cumulative some of sample index for fast acces to chunks
                        data_block['cumsum'][0] = 0
                        data_block['cumsum'][1:] = np.cumsum(data_block['size'][:-1]) // 2

                self._data_blocks[bl_type][chan_id] = data_block

        # signals channels
        sig_channels = []
        all_sig_length = []
        for chan_index in range(nb_sig_chan):
            h = slowChannelHeaders[chan_index]
            name = h['Name'].decode('utf8')
            chan_id = h['Channel']
            length = self._data_blocks[5][chan_id]['size'].sum() // 2
            if length == 0:
                continue  # channel not added
            all_sig_length.append(length)
            sampling_rate = float(h['ADFreq'])
            sig_dtype = 'int16'
            units = ''  # I dont't knwon units
            if global_header['Version'] in [100, 101]:
                gain = 5000. / (2048 * h['Gain'] * 1000.)
            elif global_header['Version'] in [102]:
                gain = 5000. / (2048 * h['Gain'] * h['PreampGain'])
            elif global_header['Version'] >= 103:
                gain = global_header['SlowMaxMagnitudeMV'] / (
                    .5 * (2 ** global_header['BitsPerSpikeSample']) *
                    h['Gain'] * h['PreampGain'])
            offset = 0.
            group_id = 0
            sig_channels.append((name, chan_id, sampling_rate, sig_dtype,
                                 units, gain, offset, group_id))
        if len(all_sig_length) > 0:
            self._signal_length = min(all_sig_length)
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        self._global_ssampling_rate = global_header['ADFrequency']
        if slowChannelHeaders.size > 0:
            assert np.unique(slowChannelHeaders['ADFreq']
                             ).size == 1, 'Signal do not have the same sampling rate'
            self._sig_sampling_rate = float(slowChannelHeaders['ADFreq'][0])

        # Determine number of units per channels
        self.internal_unit_ids = []
        for chan_id, data_clock in self._data_blocks[1].items():
            unit_ids = np.unique(data_clock['unit_id'])
            for unit_id in unit_ids:
                self.internal_unit_ids.append((chan_id, unit_id))

        # Spikes channels
        unit_channels = []
        for unit_index, (chan_id, unit_id) in enumerate(self.internal_unit_ids):
            c = np.nonzero(dspChannelHeaders['Channel'] == chan_id)[0][0]
            h = dspChannelHeaders[c]

            name = h['Name'].decode('utf8')
            _id = 'ch{}#{}'.format(chan_id, unit_id)
            wf_units = ''
            if global_header['Version'] < 103:
                wf_gain = 3000. / (2048 * h['Gain'] * 1000.)
            elif 103 <= global_header['Version'] < 105:
                wf_gain = global_header['SpikeMaxMagnitudeMV'] / (
                    .5 * 2. ** (global_header['BitsPerSpikeSample']) *
                    h['Gain'] * 1000.)
            elif global_header['Version'] >= 105:
                wf_gain = global_header['SpikeMaxMagnitudeMV'] / (
                    .5 * 2. ** (global_header['BitsPerSpikeSample']) *
                    h['Gain'] * global_header['SpikePreAmpGain'])
            wf_offset = 0.
            wf_left_sweep = -1  # DONT KNOWN
            wf_sampling_rate = global_header['WaveformFreq']
            unit_channels.append((name, _id, wf_units, wf_gain, wf_offset,
                                  wf_left_sweep, wf_sampling_rate))
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)

        # Event channels
        event_channels = []
        for chan_index in range(nb_event_chan):
            h = eventHeaders[chan_index]
            chan_id = h['Channel']
            name = h['Name'].decode('utf8')
            _id = h['Channel']
            event_channels.append((name, _id, 'event'))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        # Annotations
        self._generate_minimal_annotations()
        bl_annotations = self.raw_annotations['blocks'][0]
        seg_annotations = bl_annotations['segments'][0]
        for d in (bl_annotations, seg_annotations):
            d['rec_datetime'] = rec_datetime
            d['plexon_version'] = global_header['Version']

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop1 = float(self._last_timestamps) / self._global_ssampling_rate
        if hasattr(self, '_signal_length'):
            t_stop2 = self._signal_length / self._sig_sampling_rate
            return max(t_stop1, t_stop2)
        else:
            return t_stop1

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        return self._signal_length

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._signal_length

        if channel_indexes is None:
            channel_indexes = np.arange(self.header['signal_channels'].size)

        raw_signals = np.zeros((i_stop - i_start, len(channel_indexes)), dtype='int16')
        for c, channel_index in enumerate(channel_indexes):
            chan_header = self.header['signal_channels'][channel_index]
            chan_id = chan_header['id']

            data_blocks = self._data_blocks[5][chan_id]

            # loop over data blocks and get chunks
            bl0 = np.searchsorted(data_blocks['cumsum'], i_start, side='left')
            bl1 = np.searchsorted(data_blocks['cumsum'], i_stop, side='left')
            ind = 0
            for bl in range(bl0, bl1):
                ind0 = data_blocks[bl]['pos']
                ind1 = data_blocks[bl]['size'] + ind0
                data = self._memmap[ind0:ind1].view('int16')
                if bl == bl1 - 1:
                    # right border
                    # be carfull that bl could be both bl0 and bl1!!
                    border = data.size - (i_stop - data_blocks[bl]['cumsum'])
                    data = data[:-border]
                if bl == bl0:
                    # left border
                    border = i_start - data_blocks[bl]['cumsum']
                    data = data[border:]
                raw_signals[ind:data.size + ind, c] = data
                ind += data.size

        return raw_signals

    def _get_internal_mask(self, data_block, t_start, t_stop):
        timestamps = data_block['timestamp']

        if t_start is None:
            lim0 = 0
        else:
            lim0 = int(t_start * self._global_ssampling_rate)

        if t_stop is None:
            lim1 = self._last_timestamps
        else:
            lim1 = int(t_stop * self._global_ssampling_rate)

        keep = (timestamps >= lim0) & (timestamps <= lim1)

        return keep

    def _spike_count(self, block_index, seg_index, unit_index):
        chan_id, unit_id = self.internal_unit_ids[unit_index]
        data_block = self._data_blocks[1][chan_id]
        nb_spike = np.sum(data_block['unit_id'] == unit_id)
        return nb_spike

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        chan_id, unit_id = self.internal_unit_ids[unit_index]
        data_block = self._data_blocks[1][chan_id]

        keep = self._get_internal_mask(data_block, t_start, t_stop)
        keep &= data_block['unit_id'] == unit_id
        spike_timestamps = data_block[keep]['timestamp']

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self._global_ssampling_rate
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        chan_id, unit_id = self.internal_unit_ids[unit_index]
        data_block = self._data_blocks[1][chan_id]

        n1 = data_block['n1'][0]
        n2 = data_block['n2'][0]

        keep = self._get_internal_mask(data_block, t_start, t_stop)
        keep &= data_block['unit_id'] == unit_id

        data_block = data_block[keep]
        nb_spike = data_block.size

        waveforms = np.zeros((nb_spike, n1, n2), dtype='int16')
        for i, db in enumerate(data_block):
            ind0 = db['pos']
            ind1 = db['size'] + ind0
            data = self._memmap[ind0:ind1].view('int16').reshape(n1, n2)
            waveforms[i, :, :] = data

        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        chan_id = int(self.header['event_channels'][event_channel_index]['id'])
        nb_event = self._data_blocks[4][chan_id].size
        return nb_event

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        chan_id = int(self.header['event_channels'][event_channel_index]['id'])
        data_block = self._data_blocks[4][chan_id]
        keep = self._get_internal_mask(data_block, t_start, t_stop)

        db = data_block[keep]
        timestamps = db['timestamp']
        labels = db['label'].astype('U')
        durations = None

        return timestamps, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        event_times = event_timestamps.astype(dtype)
        event_times /= self._global_ssampling_rate
        return event_times


def read_as_dict(fid, dtype, offset=None):
    """
    Given a file descriptor
    and a numpy.dtype of the binary struct return a dict.
    Make conversion for strings.
    """
    if offset is not None:
        fid.seek(offset)
    dt = np.dtype(dtype)
    h = np.fromstring(fid.read(dt.itemsize), dt)[0]
    info = OrderedDict()
    for k in dt.names:
        v = h[k]

        if dt[k].kind == 'S':
            v = v.decode('utf8')
            v = v.replace('\x03', '')
            v = v.replace('\x00', '')

        info[k] = v
    return info


GlobalHeader = [
    ('MagicNumber', 'uint32'),
    ('Version', 'int32'),
    ('Comment', 'S128'),
    ('ADFrequency', 'int32'),
    ('NumDSPChannels', 'int32'),
    ('NumEventChannels', 'int32'),
    ('NumSlowChannels', 'int32'),
    ('NumPointsWave', 'int32'),
    ('NumPointsPreThr', 'int32'),
    ('Year', 'int32'),
    ('Month', 'int32'),
    ('Day', 'int32'),
    ('Hour', 'int32'),
    ('Minute', 'int32'),
    ('Second', 'int32'),
    ('FastRead', 'int32'),
    ('WaveformFreq', 'int32'),
    ('LastTimestamp', 'float64'),

    # version >103
    ('Trodalness', 'uint8'),
    ('DataTrodalness', 'uint8'),
    ('BitsPerSpikeSample', 'uint8'),
    ('BitsPerSlowSample', 'uint8'),
    ('SpikeMaxMagnitudeMV', 'uint16'),
    ('SlowMaxMagnitudeMV', 'uint16'),

    # version 105
    ('SpikePreAmpGain', 'uint16'),

    # version 106
    ('AcquiringSoftware', 'S18'),
    ('ProcessingSoftware', 'S18'),

    ('Padding', 'S10'),

    # all version
    ('TSCounts', 'int32', (650,)),
    ('WFCounts', 'int32', (650,)),
    ('EVCounts', 'int32', (512,)),

]

DspChannelHeader = [
    ('Name', 'S32'),
    ('SIGName', 'S32'),
    ('Channel', 'int32'),
    ('WFRate', 'int32'),
    ('SIG', 'int32'),
    ('Ref', 'int32'),
    ('Gain', 'int32'),
    ('Filter', 'int32'),
    ('Threshold', 'int32'),
    ('Method', 'int32'),
    ('NUnits', 'int32'),
    ('Template', 'uint16', (320,)),
    ('Fit', 'int32', (5,)),
    ('SortWidth', 'int32'),
    ('Boxes', 'uint16', (40,)),
    ('SortBeg', 'int32'),
    # version 105
    ('Comment', 'S128'),
    # version 106
    ('SrcId', 'uint8'),
    ('reserved', 'uint8'),
    ('ChanId', 'uint16'),

    ('Padding', 'int32', (10,)),
]

EventChannelHeader = [
    ('Name', 'S32'),
    ('Channel', 'int32'),
    # version 105
    ('Comment', 'S128'),
    # version 106
    ('SrcId', 'uint8'),
    ('reserved', 'uint8'),
    ('ChanId', 'uint16'),

    ('Padding', 'int32', (32,)),
]

SlowChannelHeader = [
    ('Name', 'S32'),
    ('Channel', 'int32'),
    ('ADFreq', 'int32'),
    ('Gain', 'int32'),
    ('Enabled', 'int32'),
    ('PreampGain', 'int32'),
    # version 104
    ('SpikeChannel', 'int32'),
    # version 105
    ('Comment', 'S128'),
    # version 106
    ('SrcId', 'uint8'),
    ('reserved', 'uint8'),
    ('ChanId', 'uint16'),

    ('Padding', 'int32', (27,)),
]

DataBlockHeader = [
    ('Type', 'uint16'),
    ('UpperByteOf5ByteTimestamp', 'uint16'),
    ('TimeStamp', 'int32'),
    ('Channel', 'uint16'),
    ('Unit', 'uint16'),
    ('NumberOfWaveforms', 'uint16'),
    ('NumberOfWordsInWaveform', 'uint16'),
]  # 16 bytes
