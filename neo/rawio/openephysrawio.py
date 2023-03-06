"""
This module implement the "old" OpenEphys format.
In this format channels are split into several files

https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Open-Ephys-format.html


Author: Samuel Garcia
"""

import os
import re

import numpy as np

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)


RECORD_SIZE = 1024
HEADER_SIZE = 1024


class OpenEphysRawIO(BaseRawIO):
    """
    OpenEphys GUI software offers several data formats, see
    https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/491632/Data+format

    This class implements the legacy OpenEphys format here
    https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/65667092/Open+Ephys+format

    The OpenEphys group already proposes some tools here:
    https://github.com/open-ephys/analysis-tools/blob/master/OpenEphys.py
    but (i) there is no package at PyPI and (ii) those tools read everything in memory.

    The format is directory based with several files:
        * .continuous
        * .events
        * .spikes

    This implementation is based on:
      * this code https://github.com/open-ephys/analysis-tools/blob/master/Python3/OpenEphys.py
        written by Dan Denman and Josh Siegle
      * a previous PR by Cristian Tatarau at Charité Berlin

    In contrast to previous code for reading this format, here all data use memmap so it should
    be super fast and light compared to legacy code.

    When the acquisition is stopped and restarted then files are named ``*_2``, ``*_3``.
    In that case this class creates a new Segment. Note that timestamps are reset in this
    situation.

    Limitation :
      * Works only if all continuous channels have the same sampling rate, which is a reasonable
        hypothesis.
      * When the recording is stopped and restarted all continuous files will contain gaps.
        Ideally this would lead to a new Segment but this use case is not implemented due to its
        complexity.
        Instead it will raise an error.

    Special cases:
      * Normally all continuous files have the same first timestamp and length. In situations
        where it is not the case all files are clipped to the smallest one so that they are all
        aligned,
        and a warning is emitted.
    """
    # file formats used by openephys
    extensions = ['continuous', 'openephys', 'spikes', 'events', 'xml']
    rawmode = 'one-dir'

    def __init__(self, dirname=''):
        BaseRawIO.__init__(self)
        self.dirname = dirname

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        info = self._info = explore_folder(self.dirname)
        nb_segment = info['nb_segment']

        # scan for continuous files
        self._sigs_memmap = {}
        self._sig_length = {}
        self._sig_timestamp0 = {}
        signal_channels = []
        oe_indices = sorted(list(info['continuous'].keys()))
        for seg_index, oe_index in enumerate(oe_indices):
            self._sigs_memmap[seg_index] = {}

            all_sigs_length = []
            all_first_timestamps = []
            all_last_timestamps = []
            all_samplerate = []
            for chan_index, continuous_filename in enumerate(info['continuous'][oe_index]):
                fullname = os.path.join(self.dirname, continuous_filename)
                chan_info = read_file_header(fullname)

                s = continuous_filename.replace('.continuous', '').split('_')
                processor_id, ch_name = s[0], s[1]
                chan_str = re.split(r'(\d+)', s[1])[0]
                # note that chan_id is not unique in case of CH + AUX
                chan_id = int(ch_name.replace(chan_str, ''))

                filesize = os.stat(fullname).st_size
                size = (filesize - HEADER_SIZE) // np.dtype(continuous_dtype).itemsize
                data_chan = np.memmap(fullname, mode='r', offset=HEADER_SIZE,
                                      dtype=continuous_dtype, shape=(size, ))
                self._sigs_memmap[seg_index][chan_index] = data_chan

                all_sigs_length.append(data_chan.size * RECORD_SIZE)
                all_first_timestamps.append(data_chan[0]['timestamp'])
                all_last_timestamps.append(data_chan[-1]['timestamp'])
                all_samplerate.append(chan_info['sampleRate'])

                # check for continuity (no gaps)
                diff = np.diff(data_chan['timestamp'])
                assert np.all(diff == RECORD_SIZE), \
                    'Not continuous timestamps for {}. ' \
                    'Maybe because recording was paused/stopped.'.format(continuous_filename)

                if seg_index == 0:
                    # add in channel list
                    if ch_name[:2].upper() == 'CH':
                        units = 'uV'
                    else:
                        units = 'V'
                    signal_channels.append((ch_name, chan_id, chan_info['sampleRate'],
                                'int16', units, chan_info['bitVolts'], 0., processor_id))

            # In some cases, continuous do not have the same length because
            # one record block is missing when the "OE GUI is freezing"
            # So we need to clip to the smallest files
            if not all(all_sigs_length[0] == e for e in all_sigs_length) or\
                    not all(all_first_timestamps[0] == e for e in all_first_timestamps):

                self.logger.warning('Continuous files do not have aligned timestamps; '
                                    'clipping to make them aligned.')

                first, last = -np.inf, np.inf
                for chan_index in self._sigs_memmap[seg_index]:
                    data_chan = self._sigs_memmap[seg_index][chan_index]
                    if data_chan[0]['timestamp'] > first:
                        first = data_chan[0]['timestamp']
                    if data_chan[-1]['timestamp'] < last:
                        last = data_chan[-1]['timestamp']

                all_sigs_length = []
                all_first_timestamps = []
                all_last_timestamps = []
                for chan_index in self._sigs_memmap[seg_index]:
                    data_chan = self._sigs_memmap[seg_index][chan_index]
                    keep = (data_chan['timestamp'] >= first) & (data_chan['timestamp'] <= last)
                    data_chan = data_chan[keep]
                    self._sigs_memmap[seg_index][chan_index] = data_chan
                    all_sigs_length.append(data_chan.size * RECORD_SIZE)
                    all_first_timestamps.append(data_chan[0]['timestamp'])
                    all_last_timestamps.append(data_chan[-1]['timestamp'])

            # check that all signals have the same length and timestamp0 for this segment
            assert all(all_sigs_length[0] == e for e in all_sigs_length),\
                       'Not all signals have the same length'
            assert all(all_first_timestamps[0] == e for e in all_first_timestamps),\
                       'Not all signals have the same first timestamp'
            assert all(all_samplerate[0] == e for e in all_samplerate),\
                       'Not all signals have the same sample rate'

            self._sig_length[seg_index] = all_sigs_length[0]
            self._sig_timestamp0[seg_index] = all_first_timestamps[0]

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        self._sig_sampling_rate = signal_channels['sampling_rate'][0]  # unique for channel

        # split channels in stream depending the name CHxxx ADCxxx
        chan_stream_ids = [name[:2] if name.startswith('CH') else name[:3]
                      for name in signal_channels['name']]
        signal_channels['stream_id'] = chan_stream_ids

        # and create streams channels (keep natural order 'CH' first)
        stream_ids, order = np.unique(chan_stream_ids, return_index=True)
        stream_ids = stream_ids[np.argsort(order)]
        signal_streams = [(f'Signals {stream_id}', f'{stream_id}') for stream_id in stream_ids]
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        # scan for spikes files
        spike_channels = []

        if len(info['spikes']) > 0:

            self._spikes_memmap = {}
            for seg_index, oe_index in enumerate(oe_indices):
                self._spikes_memmap[seg_index] = {}
                for spike_filename in info['spikes'][oe_index]:
                    fullname = os.path.join(self.dirname, spike_filename)
                    spike_info = read_file_header(fullname)
                    spikes_dtype = make_spikes_dtype(fullname)

                    # "STp106.0n0_2.spikes" to "STp106.0n0"
                    name = spike_filename.replace('.spikes', '')
                    if seg_index > 0:
                        name = name.replace('_' + str(seg_index + 1), '')

                    data_spike = np.memmap(fullname, mode='r', offset=HEADER_SIZE,
                                        dtype=spikes_dtype)
                    self._spikes_memmap[seg_index][name] = data_spike

            # In each file 'sorted_id' indicate the number of cluster so number of units
            # so need to scan file for all segment to get units
            self._spike_sampling_rate = None
            for spike_filename_seg0 in info['spikes'][0]:
                name = spike_filename_seg0.replace('.spikes', '')

                fullname = os.path.join(self.dirname, spike_filename_seg0)
                spike_info = read_file_header(fullname)
                if self._spike_sampling_rate is None:
                    self._spike_sampling_rate = spike_info['sampleRate']
                else:
                    assert self._spike_sampling_rate == spike_info['sampleRate'],\
                        'mismatch in spike sampling rate'

                # scan all to detect several all unique(sorted_ids)
                all_sorted_ids = []
                for seg_index in range(nb_segment):
                    data_spike = self._spikes_memmap[seg_index][name]
                    all_sorted_ids += np.unique(data_spike['sorted_id']).tolist()
                all_sorted_ids = np.unique(all_sorted_ids)

                # suppose all channels have the same gain
                wf_units = 'uV'
                wf_gain = 1000. / data_spike[0]['gains'][0]
                wf_offset = - (2**15) * wf_gain
                wf_left_sweep = 0
                wf_sampling_rate = spike_info['sampleRate']

                # each sorted_id is one channel
                for sorted_id in all_sorted_ids:
                    unit_name = "{}#{}".format(name, sorted_id)
                    unit_id = "{}#{}".format(name, sorted_id)
                    spike_channels.append((unit_name, unit_id, wf_units,
                                wf_gain, wf_offset, wf_left_sweep, wf_sampling_rate))

        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # event file are:
        #    * all_channel.events (header + binray)  -->  event 0
        # and message.events (text based)      --> event 1 not implemented yet
        event_channels = []
        self._events_memmap = {}
        for seg_index, oe_index in enumerate(oe_indices):
            if oe_index == 0:
                event_filename = 'all_channels.events'
            else:
                event_filename = 'all_channels_{}.events'.format(oe_index + 1)

            fullname = os.path.join(self.dirname, event_filename)
            event_info = read_file_header(fullname)
            self._event_sampling_rate = event_info['sampleRate']
            data_event = np.memmap(fullname, mode='r', offset=HEADER_SIZE,
                                   dtype=events_dtype)
            self._events_memmap[seg_index] = data_event

        event_channels.append(('all_channels', '', 'event'))
        # event_channels.append(('message', '', 'event')) # not implemented
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # main header
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [nb_segment]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        # Annotate some objects from continuous files
        self._generate_minimal_annotations()
        bl_ann = self.raw_annotations['blocks'][0]
        for seg_index, oe_index in enumerate(oe_indices):
            seg_ann = bl_ann['segments'][seg_index]
            if len(info['continuous']) > 0:
                fullname = os.path.join(self.dirname, info['continuous'][oe_index][0])
                chan_info = read_file_header(fullname)
                seg_ann['openephys_version'] = chan_info['version']
                bl_ann['openephys_version'] = chan_info['version']
                seg_ann['date_created'] = chan_info['date_created']
                seg_ann['openephys_segment_index'] = oe_index + 1

    def _segment_t_start(self, block_index, seg_index):
        # segment start/stop are defined by continuous channels
        return self._sig_timestamp0[seg_index] / self._sig_sampling_rate

    def _segment_t_stop(self, block_index, seg_index):
        return (self._sig_timestamp0[seg_index] + self._sig_length[seg_index])\
            / self._sig_sampling_rate

    def _get_signal_size(self, block_index, seg_index, stream_index):
        return self._sig_length[seg_index]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return self._sig_timestamp0[seg_index] / self._sig_sampling_rate

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._sig_length[seg_index]

        block_start = i_start // RECORD_SIZE
        block_stop = i_stop // RECORD_SIZE + 1
        sl0 = i_start % RECORD_SIZE
        sl1 = sl0 + (i_stop - i_start)

        stream_id = self.header['signal_streams'][stream_index]['id']
        mask = self.header['signal_channels']['stream_id']
        global_channel_indexes, = np.nonzero(mask == stream_id)
        if channel_indexes is None:
            channel_indexes = slice(None)
        global_channel_indexes = global_channel_indexes[channel_indexes]

        sigs_chunk = np.zeros((i_stop - i_start, len(global_channel_indexes)), dtype='int16')
        for i, global_chan_index in enumerate(global_channel_indexes):
            data = self._sigs_memmap[seg_index][global_chan_index]
            sub = data[block_start:block_stop]
            sigs_chunk[:, i] = sub['samples'].flatten()[sl0:sl1]

        return sigs_chunk

    def _get_spike_slice(self, seg_index, unit_index, t_start, t_stop):
        name, sorted_id = self.header['spike_channels'][unit_index]['name'].split('#')
        sorted_id = int(sorted_id)
        data_spike = self._spikes_memmap[seg_index][name]

        if t_start is None:
            t_start = self._segment_t_start(0, seg_index)
        if t_stop is None:
            t_stop = self._segment_t_stop(0, seg_index)
        ts0 = int(t_start * self._spike_sampling_rate)
        ts1 = int(t_stop * self._spike_sampling_rate)

        ts = data_spike['timestamp']
        keep = (data_spike['sorted_id'] == sorted_id) & (ts >= ts0) & (ts <= ts1)
        return data_spike, keep

    def _spike_count(self, block_index, seg_index, unit_index):
        data_spike, keep = self._get_spike_slice(seg_index, unit_index, None, None)
        return np.sum(keep)

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        data_spike, keep = self._get_spike_slice(seg_index, unit_index, t_start, t_stop)
        return data_spike['timestamp'][keep]

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype) / self._spike_sampling_rate
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        data_spike, keep = self._get_spike_slice(seg_index, unit_index, t_start, t_stop)
        nb_chan = data_spike[0]['nb_channel']
        nb = np.sum(keep)
        waveforms = data_spike[keep]['samples'].flatten()
        waveforms = waveforms.reshape(nb, nb_chan, -1)
        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        # assert event_channel_index==0
        return self._events_memmap[seg_index].size

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        # assert event_channel_index==0

        if t_start is None:
            t_start = self._segment_t_start(block_index, seg_index)
        if t_stop is None:
            t_stop = self._segment_t_stop(block_index, seg_index)
        ts0 = int(t_start * self._event_sampling_rate)
        ts1 = int(t_stop * self._event_sampling_rate)
        ts = self._events_memmap[seg_index]['timestamp']
        keep = (ts >= ts0) & (ts <= ts1)

        subdata = self._events_memmap[seg_index][keep]
        timestamps = subdata['timestamp']
        # question what is the label????
        # here I put a combination
        labels = np.array(['{}#{}#{}'.format(int(d['event_type']),
                                             int(d['processor_id']),
                                             int(d['chan_id']))
                           for d in subdata],
                          dtype='U')
        durations = None

        return timestamps, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        event_times = event_timestamps.astype(dtype) / self._event_sampling_rate
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        return None


continuous_dtype = [('timestamp', 'int64'), ('nb_sample', 'uint16'),
    ('rec_num', 'uint16'), ('samples', '>i2', RECORD_SIZE),
    ('markers', 'uint8', 10)]

events_dtype = [('timestamp', 'int64'), ('sample_pos', 'int16'),
    ('event_type', 'uint8'), ('processor_id', 'uint8'),
    ('event_id', 'uint8'), ('chan_id', 'uint8'),
    ('record_num', 'uint16')]

# the dtype is dynamic and depend on nb_channel and nb_sample
_base_spikes_dtype = [('event_stype', 'uint8'), ('timestamp', 'int64'),
    ('software_timestamp', 'int64'), ('source_id', 'uint16'),
    ('nb_channel', 'uint16'), ('nb_sample', 'uint16'),
    ('sorted_id', 'uint16'), ('electrode_id', 'uint16'),
    ('within_chan_index', 'uint16'), ('color', 'uint8', 3),
    ('pca', 'float32', 2), ('sampling_rate', 'uint16'),
    ('samples', 'uint16', None), ('gains', 'float32', None),
    ('thresholds', 'uint16', None), ('rec_num', 'uint16')]


def make_spikes_dtype(filename):
    """
    Given the spike file make the appropriate dtype that depends on:
      * N - number of channels
      * M - samples per spike
    See documentation of file format.
    """

    # strangely the header do not have the sample size
    # So this do not work (too bad):
    # spike_info = read_file_header(filename)
    # N = spike_info['num_channels']
    # M =????

    # so we need to read the very first spike
    # but it will fail when 0 spikes (too bad)
    filesize = os.stat(filename).st_size
    if filesize >= (HEADER_SIZE + 23):
        with open(filename, mode='rb') as f:
            # M and N is at 1024 + 19 bytes
            f.seek(HEADER_SIZE + 19)
            N = np.fromfile(f, np.dtype('<u2'), 1)[0]
            M = np.fromfile(f, np.dtype('<u2'), 1)[0]
    else:
        spike_info = read_file_header(filename)
        N = spike_info['num_channels']
        M = 40  # this is in the original code from openephys

    # make a copy
    spikes_dtype = [e for e in _base_spikes_dtype]
    spikes_dtype[12] = ('samples', 'uint16', N * M)
    spikes_dtype[13] = ('gains', 'float32', N)
    spikes_dtype[14] = ('thresholds', 'uint16', N)

    return spikes_dtype


def explore_folder(dirname):
    """
    This explores a folder and dispatch continuous, event and spikes
    files by segment (aka recording session).

    The number of segments is checked with these rules
    "100_CH0.continuous" ---> seg_index 0
    "100_CH0_2.continuous" ---> seg_index 1
    "100_CH0_N.continuous" ---> seg_index N-1
    """
    filenames = os.listdir(dirname)
    filenames.sort()

    info = {}
    info['nb_segment'] = 0
    info['continuous'] = {}
    info['spikes'] = {}
    for filename in filenames:
        if filename.endswith('.continuous'):
            s = filename.replace('.continuous', '').split('_')
            if len(s) == 2:
                seg_index = 0
            else:
                seg_index = int(s[2]) - 1
            if seg_index not in info['continuous'].keys():
                info['continuous'][seg_index] = []
            info['continuous'][seg_index].append(filename)
            if (seg_index + 1) > info['nb_segment']:
                info['nb_segment'] += 1
        elif filename.endswith('.spikes'):
            s = filename.replace('.spikes', '').split('_')
            if len(s) == 1:
                seg_index = 0
            else:
                seg_index = int(s[1]) - 1
            if seg_index not in info['spikes'].keys():
                info['spikes'][seg_index] = []
            info['spikes'][seg_index].append(filename)
            if (seg_index + 1) > info['nb_segment']:
                info['nb_segment'] += 1

    # order continuous file by channel number within segment
    # order "CH before "ADC"
    for seg_index, continuous_filenames in info['continuous'].items():
        chan_ids_by_type = {}
        filenames_by_type = {}
        for continuous_filename in continuous_filenames:
            s = continuous_filename.replace('.continuous', '').split('_')
            processor_id, ch_name = s[0], s[1]
            chan_type = re.split(r'(\d+)', s[1])[0]
            chan_id = int(ch_name.replace(chan_type, ''))
            if chan_type in chan_ids_by_type.keys():
                chan_ids_by_type[chan_type].append(chan_id)
                filenames_by_type[chan_type].append(continuous_filename)
            else:
                chan_ids_by_type[chan_type] = [chan_id]
                filenames_by_type[chan_type] = [continuous_filename]
        chan_types = list(chan_ids_by_type.keys())
        if chan_types[0] == 'ADC':
            # put ADC at last position
            chan_types = chan_types[1:] + chan_types[0:1]
        ordered_continuous_filenames = []
        for chan_type in chan_types:
            local_order = np.argsort(chan_ids_by_type[chan_type])
            local_filenames = np.array(filenames_by_type[chan_type])[local_order]
            ordered_continuous_filenames.extend(local_filenames)
        info['continuous'][seg_index] = ordered_continuous_filenames

    # order spike files within segment
    for seg_index, spike_filenames in info['spikes'].items():
        names = []
        for spike_filename in spike_filenames:
            name = spike_filename.replace('.spikes', '')
            if seg_index > 0:
                name = name.replace('_' + str(seg_index + 1), '')
            names.append(name)
        order = np.argsort(names)
        spike_filenames = [spike_filenames[i] for i in order]
        info['spikes'][seg_index] = spike_filenames

    return info


def read_file_header(filename):
    """Read header information from the first 1024 bytes of an OpenEphys file.
    See docs.
    """
    header = {}
    with open(filename, mode='rb') as f:
        # Read the data as a string
        # Remove newlines and redundant "header." prefixes
        # The result should be a series of "key = value" strings, separated
        # by semicolons.
        header_string = f.read(HEADER_SIZE).replace(b'\n', b'').replace(b'header.', b'')

    # Parse each key = value string separately
    for pair in header_string.split(b';'):
        if b'=' in pair:
            key, value = pair.split(b' = ')
            key = key.strip().decode('ascii')
            value = value.strip()

            # Convert some values to numeric
            if key in ['bitVolts', 'sampleRate']:
                header[key] = float(value)
            elif key in ['blockLength', 'bufferSize', 'header_bytes', 'num_channels']:
                header[key] = int(value)
            else:
                # Keep as string
                header[key] = value.decode('ascii')

    return header
