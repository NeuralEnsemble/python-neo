"""
Class for reading data from BrainVision product.

This code was originally made by L. Pezard (2010), modified B. Burle and
S. More.

Author: Samuel Garcia
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)

import numpy as np

import datetime
import os
import re


class BrainVisionRawIO(BaseRawIO):
    """

    """
    extensions = ['vhdr']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _parse_header(self):
        # Read header file (vhdr)
        vhdr_header = read_brainvsion_soup(self.filename)

        bname = os.path.basename(self.filename)
        marker_filename = self.filename.replace(bname, vhdr_header['Common Infos']['MarkerFile'])
        binary_filename = self.filename.replace(bname, vhdr_header['Common Infos']['DataFile'])

        assert vhdr_header['Common Infos'][
            'DataFormat'] == 'BINARY', NotImplementedError
        assert vhdr_header['Common Infos'][
            'DataOrientation'] == 'MULTIPLEXED', NotImplementedError

        nb_channel = int(vhdr_header['Common Infos']['NumberOfChannels'])
        sr = 1.e6 / float(vhdr_header['Common Infos']['SamplingInterval'])
        self._sampling_rate = sr

        fmt = vhdr_header['Binary Infos']['BinaryFormat']
        fmts = {'INT_16': np.int16, 'INT_32': np.int32, 'IEEE_FLOAT_32': np.float32, }

        assert fmt in fmts, NotImplementedError
        sig_dtype = fmts[fmt]

        # raw signals memmap
        sigs = np.memmap(binary_filename, dtype=sig_dtype, mode='r', offset=0)
        if sigs.size % nb_channel != 0:
            sigs = sigs[:-sigs.size % nb_channel]
        self._raw_signals = sigs.reshape(-1, nb_channel)

        signal_streams = np.array([('Signals', '0')], dtype=_signal_stream_dtype)

        sig_channels = []
        channel_infos = vhdr_header['Channel Infos']
        for c in range(nb_channel):
            try:
                channel_desc = channel_infos['Ch%d' % (c + 1,)]
            except KeyError:
                channel_desc = channel_infos['ch%d' % (c + 1,)]
            name, ref, res, units = channel_desc.split(',')
            units = units.replace('µ', 'u')
            chan_id = str(c + 1)
            if sig_dtype == np.int16 or sig_dtype == np.int32:
                gain = float(res)
            else:
                gain = 1
            offset = 0
            stream_id = '0'
            sig_channels.append((name, chan_id, self._sampling_rate, sig_dtype,
                                 units, gain, offset, stream_id))
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # read all markers in memory

        all_info = read_brainvsion_soup(marker_filename)['Marker Infos']
        ev_types = []
        ev_timestamps = []
        ev_labels = []
        for i in range(len(all_info)):
            ev_type, ev_label, pos, size, channel = all_info[
                'Mk%d' % (i + 1,)].split(',')[:5]
            ev_types.append(ev_type)
            ev_timestamps.append(int(pos))
            ev_labels.append(ev_label)
        ev_types = np.array(ev_types)
        ev_timestamps = np.array(ev_timestamps)
        ev_labels = np.array(ev_labels, dtype='U')

        # group them by types
        self._raw_events = []
        event_channels = []
        for c, ev_type in enumerate(np.unique(ev_types)):
            ind = (ev_types == ev_type)
            event_channels.append((ev_type, '', 'event'))

            self._raw_events.append((ev_timestamps[ind], ev_labels[ind]))

        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = sig_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()
        if 'Coordinates' in vhdr_header:
            sig_annotations = self.raw_annotations['blocks'][0]['segments'][0]['signals'][0]
            all_coords = []
            for c in range(sig_channels.size):
                coords = vhdr_header['Coordinates']['Ch{}'.format(c + 1)]
                all_coords.append([float(v) for v in coords.split(',')])
            all_coords = np.array(all_coords)
            for dim in range(all_coords.shape[1]):
                sig_annotations['__array_annotations__'][f'coordinates_{dim}'] = all_coords[:, dim]

    def _source_name(self):
        return self.filename

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._raw_signals.shape[0] / self._sampling_rate
        return t_stop

    ###
    def _get_signal_size(self, block_index, seg_index, stream_index):
        assert stream_index == 0
        return self._raw_signals.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_signals[slice(i_start, i_stop), channel_indexes]
        return raw_signals

    ###
    def _spike_count(self, block_index, seg_index, unit_index):
        return 0

    ###
    # event and epoch zone
    def _event_count(self, block_index, seg_index, event_channel_index):
        all_timestamps, all_label = self._raw_events[event_channel_index]
        return all_timestamps.size

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        timestamps, labels = self._raw_events[event_channel_index]

        if t_start is not None:
            keep = timestamps >= int(t_start * self._sampling_rate)
            timestamps = timestamps[keep]
            labels = labels[keep]

        if t_stop is not None:
            keep = timestamps <= int(t_stop * self._sampling_rate)
            timestamps = timestamps[keep]
            labels = labels[keep]

        durations = None

        return timestamps, durations, labels

        raise (NotImplementedError)

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        event_times = event_timestamps.astype(dtype) / self._sampling_rate
        return event_times


def read_brainvsion_soup(filename):
    with open(filename, 'r', encoding='utf8') as f:
        section = None
        all_info = {}
        for line in f:
            line = line.strip('\n').strip('\r')
            if line.startswith('['):
                section = re.findall(r'\[([\S ]+)\]', line)[0]
                all_info[section] = {}
                continue
            if line.startswith(';'):
                continue
            if '=' in line and len(line.split('=')) == 2:
                k, v = line.split('=')
                all_info[section][k] = v

    return all_info
