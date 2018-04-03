# -*- coding: utf-8 -*-
"""
Class for reading data from Neuralynx files.
This IO supports NCS, NEV, NSE and NTT file formats.


NCS contains signals for one channel
NEV contains events
NSE contains spikes and waveforms for mono electrodes
NTT contains spikes and waveforms for tetrodes


NCS can contains gaps that can be detected in inregularity
in timestamps of data blocks. Each gap lead to one new segment.
NCS files need to be read entirely to detect that gaps.... too bad....


Author: Julia Sprenger, Carlos Canova, Samuel Garcia
"""
from __future__ import print_function, division, absolute_import
# from __future__ import unicode_literals is not compatible with numpy.dtype both py2 py3


from .baserawio import (BaseRawIO, _signal_channel_dtype,
                        _unit_channel_dtype, _event_channel_dtype)

import numpy as np
import os
import re
import distutils.version
import datetime
from collections import OrderedDict

BLOCK_SIZE = 512  # nb sample per signal block
HEADER_SIZE = 2 ** 14  # file have a txt header of 16kB


class NeuralynxRawIO(BaseRawIO):
    """"
    Class for reading dataset recorded by Neuralynx.

    Examples:
        >>> reader = NeuralynxRawIO(dirname='Cheetah_v5.5.1/original_data')
        >>> reader.parse_header()

            Inspect all file in the directory.

        >>> print(reader)

            Display all informations about signal channels, units, segment size....
    """
    extensions = ['nse', 'ncs', 'nev', 'ntt']
    rawmode = 'one-dir'

    def __init__(self, dirname='', **kargs):
        self.dirname = dirname
        BaseRawIO.__init__(self, **kargs)

    def _source_name(self):
        return self.dirname

    def _parse_header(self):

        sig_channels = []
        unit_channels = []
        event_channels = []

        self.ncs_filenames = OrderedDict()  # chan_id: filename
        self.nse_ntt_filenames = OrderedDict()  # chan_id: filename
        self.nev_filenames = OrderedDict()  # chan_id: filename

        self._nev_memmap = {}
        self._spike_memmap = {}
        self.internal_unit_ids = []  # channel_index > (channel_id, unit_id)
        self.internal_event_ids = []

        # explore the directory looking for ncs, nev, nse and ntt
        # And construct channels headers
        signal_annotations = []
        unit_annotations = []
        event_annotations = []
        for filename in sorted(os.listdir(self.dirname)):
            filename = os.path.join(self.dirname, filename)

            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # remove dot
            if ext not in self.extensions:
                continue

            # All file have more or less the same header structure
            info = read_txt_header(filename)
            chan_names = info['channel_names']
            chan_ids = info['channel_ids']

            for idx, chan_id in enumerate(chan_ids):
                chan_name = chan_names[idx]
                if ext == 'ncs':
                    # a signal channels
                    units = 'uV'
                    gain = info['bit_to_microVolt'][idx]
                    if info['input_inverted']:
                        gain *= -1
                    offset = 0.
                    group_id = 0
                    sig_channels.append((chan_name, chan_id, info['sampling_rate'],
                                         'int16', units, gain, offset, group_id))
                    self.ncs_filenames[chan_id] = filename
                    keys = [
                        'DspFilterDelay_µs',
                        'recording_opened',
                        'FileType',
                        'DspDelayCompensation',
                        'recording_closed',
                        'DspLowCutFilterType',
                        'HardwareSubSystemName',
                        'DspLowCutNumTaps',
                        'DSPLowCutFilterEnabled',
                        'HardwareSubSystemType',
                        'DspHighCutNumTaps',
                        'ADMaxValue',
                        'DspLowCutFrequency',
                        'DSPHighCutFilterEnabled',
                        'RecordSize',
                        'InputRange',
                        'DspHighCutFrequency',
                        'input_inverted',
                        'NumADChannels',
                        'DspHighCutFilterType',
                    ]
                    d = {k: info[k] for k in keys if k in info}
                    signal_annotations.append(d)

                elif ext in ('nse', 'ntt'):
                    # nse and ntt are pretty similar except for the wavform shape
                    # a file can contain several unit_id (so several unit channel)
                    assert chan_id not in self.nse_ntt_filenames, \
                        'Several nse or ntt files have the same unit_id!!!'
                    self.nse_ntt_filenames[chan_id] = filename

                    dtype = get_nse_or_ntt_dtype(info, ext)
                    data = np.memmap(filename, dtype=dtype, mode='r', offset=HEADER_SIZE)
                    self._spike_memmap[chan_id] = data

                    unit_ids = np.unique(data['unit_id'])
                    for unit_id in unit_ids:
                        # a spike channel for each (chan_id, unit_id)
                        self.internal_unit_ids.append((chan_id, unit_id))

                        unit_name = "ch{}#{}".format(chan_id, unit_id)
                        unit_id = '{}'.format(unit_id)
                        wf_units = 'uV'
                        wf_gain = info['bit_to_microVolt'][idx]
                        if info['input_inverted']:
                            wf_gain *= -1
                        wf_offset = 0.
                        wf_left_sweep = -1  # NOT KNOWN
                        wf_sampling_rate = info['sampling_rate']
                        unit_channels.append(
                            (unit_name, '{}'.format(unit_id), wf_units, wf_gain,
                             wf_offset, wf_left_sweep, wf_sampling_rate))
                        unit_annotations.append(dict(file_origin=filename))

                elif ext == 'nev':
                    # an event channel
                    # each ('event_id',  'ttl_input') give a new event channel
                    self.nev_filenames[chan_id] = filename
                    data = np.memmap(
                        filename, dtype=nev_dtype, mode='r', offset=HEADER_SIZE)
                    internal_ids = np.unique(
                        data[['event_id', 'ttl_input']]).tolist()
                    for internal_event_id in internal_ids:
                        if internal_event_id not in self.internal_event_ids:
                            event_id, ttl_input = internal_event_id
                            name = '{} event_id={} ttl={}'.format(
                                chan_name, event_id, ttl_input)
                            event_channels.append((name, chan_id, 'event'))
                            self.internal_event_ids.append(internal_event_id)

                    self._nev_memmap[chan_id] = data

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        if sig_channels.size > 0:
            sampling_rate = np.unique(sig_channels['sampling_rate'])
            assert sampling_rate.size == 1
            self._sigs_sampling_rate = sampling_rate[0]

        # read ncs files for gaps detection and nb_segment computation
        self.read_ncs_files(self.ncs_filenames)

        # timestamp limit in nev, nse
        # so need to scan all spike and event to
        ts0, ts1 = None, None
        for _data_memmap in (self._spike_memmap, self._nev_memmap):
            for chan_id, data in _data_memmap.items():
                ts = data['timestamp']
                if ts.size == 0:
                    continue
                if ts0 is None:
                    ts0 = ts[0]
                    ts1 = ts[-1]
                ts0 = min(ts0, ts[0])
                ts1 = max(ts0, ts[-1])

        if self._timestamp_limits is None:
            # case  NO ncs but HAVE nev or nse
            self._timestamp_limits = [(ts0, ts1)]
            self._seg_t_starts = [ts0 / 1e6]
            self._seg_t_stops = [ts1 / 1e6]
            self.global_t_start = ts0 / 1e6
            self.global_t_stop = ts1 / 1e6
        elif ts0 is not None:
            # case  HAVE ncs AND HAVE nev or nse
            self.global_t_start = min(ts0 / 1e6, self._sigs_t_start[0])
            self.global_t_stop = max(ts1 / 1e6, self._sigs_t_stop[-1])
            self._seg_t_starts = list(self._sigs_t_start)
            self._seg_t_starts[0] = self.global_t_start
            self._seg_t_stops = list(self._sigs_t_stop)
            self._seg_t_stops[-1] = self.global_t_stop
        else:
            # case HAVE ncs but  NO nev or nse
            self._seg_t_starts = self._sigs_t_start
            self._seg_t_stops = self._sigs_t_stop
            self.global_t_start = self._sigs_t_start[0]
            self.global_t_stop = self._sigs_t_stop[-1]

        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [self._nb_segment]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        # Annotations
        self._generate_minimal_annotations()
        bl_annotations = self.raw_annotations['blocks'][0]

        for seg_index in range(self._nb_segment):
            seg_annotations = bl_annotations['segments'][seg_index]

            for c in range(sig_channels.size):
                sig_ann = seg_annotations['signals'][c]
                sig_ann.update(signal_annotations[c])

            for c in range(unit_channels.size):
                unit_ann = seg_annotations['units'][c]
                unit_ann.update(unit_annotations[c])

            for c in range(event_channels.size):
                # annotations for channel events
                event_id, ttl_input = self.internal_event_ids[c]
                chan_id = event_channels[c]['id']

                ev_ann = seg_annotations['events'][c]
                ev_ann['file_origin'] = self.nev_filenames[chan_id]

                # ~ ev_ann['marker_id'] =
                # ~ ev_ann['nttl'] =
                # ~ ev_ann['digital_marker'] =
                # ~ ev_ann['analog_marker'] =

    def _segment_t_start(self, block_index, seg_index):
        return self._seg_t_starts[seg_index] - self.global_t_start

    def _segment_t_stop(self, block_index, seg_index):
        return self._seg_t_stops[seg_index] - self.global_t_start

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        return self._sigs_length[seg_index]

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return self._sigs_t_start[seg_index] - self.global_t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._sigs_length[seg_index]

        block_start = i_start // BLOCK_SIZE
        block_stop = i_stop // BLOCK_SIZE + 1
        sl0 = i_start % 512
        sl1 = sl0 + (i_stop - i_start)

        if channel_indexes is None:
            channel_indexes = slice(None)
        channel_ids = self.header['signal_channels'][channel_indexes]['id']

        sigs_chunk = np.zeros((i_stop - i_start, len(channel_ids)), dtype='int16')
        for i, chan_id in enumerate(channel_ids):
            data = self._sigs_memmap[seg_index][chan_id]
            sub = data[block_start:block_stop]
            sigs_chunk[:, i] = sub['samples'].flatten()[sl0:sl1]

        return sigs_chunk

    def _spike_count(self, block_index, seg_index, unit_index):
        chan_id, unit_id = self.internal_unit_ids[unit_index]
        data = self._spike_memmap[chan_id]
        ts = data['timestamp']

        ts0, ts1 = self._timestamp_limits[seg_index]

        keep = (ts >= ts0) & (ts <= ts1) & (unit_id == data['unit_id'])
        nb_spike = int(data[keep].size)
        return nb_spike

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        chan_id, unit_id = self.internal_unit_ids[unit_index]
        data = self._spike_memmap[chan_id]
        ts = data['timestamp']

        ts0, ts1 = self._timestamp_limits[seg_index]
        if t_start is not None:
            ts0 = int((t_start + self.global_t_start) * 1e6)
        if t_start is not None:
            ts1 = int((t_stop + self.global_t_start) * 1e6)

        keep = (ts >= ts0) & (ts <= ts1) & (unit_id == data['unit_id'])
        timestamps = ts[keep]
        return timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= 1e6
        spike_times -= self.global_t_start
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index,
                                 t_start, t_stop):
        chan_id, unit_id = self.internal_unit_ids[unit_index]
        data = self._spike_memmap[chan_id]
        ts = data['timestamp']

        ts0, ts1 = self._timestamp_limits[seg_index]
        if t_start is not None:
            ts0 = int((t_start + self.global_t_start) * 1e6)
        if t_start is not None:
            ts1 = int((t_stop + self.global_t_start) * 1e6)

        keep = (ts >= ts0) & (ts <= ts1) & (unit_id == data['unit_id'])

        wfs = data[keep]['samples']
        if wfs.ndim == 2:
            # case for nse
            waveforms = wfs[:, None, :]
        else:
            # case for ntt change (n, 32, 4) to (n, 4, 32)
            waveforms = wfs.swapaxes(1, 2)

        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        event_id, ttl_input = self.internal_event_ids[event_channel_index]
        chan_id = self.header['event_channels'][event_channel_index]['id']
        data = self._nev_memmap[chan_id]
        ts0, ts1 = self._timestamp_limits[seg_index]
        ts = data['timestamp']
        keep = (ts >= ts0) & (ts <= ts1) & (data['event_id'] == event_id) & \
               (data['ttl_input'] == ttl_input)
        nb_event = int(data[keep].size)
        return nb_event

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        event_id, ttl_input = self.internal_event_ids[event_channel_index]
        chan_id = self.header['event_channels'][event_channel_index]['id']
        data = self._nev_memmap[chan_id]
        ts0, ts1 = self._timestamp_limits[seg_index]

        if t_start is not None:
            ts0 = int((t_start + self.global_t_start) * 1e6)
        if t_start is not None:
            ts1 = int((t_stop + self.global_t_start) * 1e6)

        ts = data['timestamp']
        keep = (ts >= ts0) & (ts <= ts1) & (data['event_id'] == event_id) & \
               (data['ttl_input'] == ttl_input)

        subdata = data[keep]
        timestamps = subdata['timestamp']
        labels = subdata['event_string'].astype('U')
        durations = None
        return timestamps, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        event_times = event_timestamps.astype(dtype)
        event_times /= 1e6
        event_times -= self.global_t_start
        return event_times

    def read_ncs_files(self, ncs_filenames):
        """
        Given a list of ncs files contrsuct:
            * self._sigs_memmap = [ {} for seg_index in range(self._nb_segment) ]
            * self._sigs_t_start = []
            * self._sigs_t_stop = []
            * self._sigs_length = []
            * self._nb_segment
            * self._timestamp_limits

        The first file is read entirely to detect gaps in timestamp.
        each gap lead to a new segment.

        Other files are not read entirely but we check than gaps
        are at the same place.


        gap_indexes can be given (when cached) to avoid full read.

        """
        if len(ncs_filenames) == 0:
            self._nb_segment = 1
            self._timestamp_limits = None
            return

        good_delta = int(BLOCK_SIZE * 1e6 / self._sigs_sampling_rate)
        chan_id0 = list(ncs_filenames.keys())[0]
        filename0 = ncs_filenames[chan_id0]

        data0 = np.memmap(filename0, dtype=ncs_dtype, mode='r', offset=HEADER_SIZE)

        gap_indexes = None
        if self.use_cache:
            gap_indexes = self._cache.get('gap_indexes')

        # detect gaps on first file
        if gap_indexes is None:
            # this can be long!!!!
            timestamps0 = data0['timestamp']
            deltas0 = np.diff(timestamps0)

            # It should be that:
            # gap_indexes, = np.nonzero(deltas0!=good_delta)
            # but for a file I have found many deltas0==15999 deltas0==16000
            # I guess this is a round problem
            # So this is the same with a tolerance of 1 or 2 ticks
            mask = deltas0 != good_delta
            for tolerance in (1, 2):
                mask &= (deltas0 != good_delta - tolerance)
                mask &= (deltas0 != good_delta + tolerance)
            gap_indexes, = np.nonzero(mask)

            if self.use_cache:
                self.add_in_cache(gap_indexes=gap_indexes)

        gap_bounds = [0] + (gap_indexes + 1).tolist() + [data0.size]
        self._nb_segment = len(gap_bounds) - 1

        self._sigs_memmap = [{} for seg_index in range(self._nb_segment)]
        self._sigs_t_start = []
        self._sigs_t_stop = []
        self._sigs_length = []
        self._timestamp_limits = []
        # create segment with subdata block/t_start/t_stop/length
        for chan_id, ncs_filename in self.ncs_filenames.items():
            data = np.memmap(ncs_filename, dtype=ncs_dtype, mode='r', offset=HEADER_SIZE)
            assert data.size == data0.size, 'ncs files do not have the same data length'

            for seg_index in range(self._nb_segment):
                i0 = gap_bounds[seg_index]
                i1 = gap_bounds[seg_index + 1]

                assert data[i0]['timestamp'] == data0[i0][
                    'timestamp'], 'ncs files do not have the same gaps'
                assert data[i1 - 1]['timestamp'] == data0[i1 - 1][
                    'timestamp'], 'ncs files do not have the same gaps'

                subdata = data[i0:i1]
                self._sigs_memmap[seg_index][chan_id] = subdata

                if chan_id == chan_id0:
                    ts0 = subdata[0]['timestamp']
                    ts1 = subdata[-1]['timestamp'] + \
                          np.uint64(BLOCK_SIZE / self._sigs_sampling_rate * 1e6)
                    self._timestamp_limits.append((ts0, ts1))
                    t_start = ts0 / 1e6
                    self._sigs_t_start.append(t_start)
                    t_stop = ts1 / 1e6
                    self._sigs_t_stop.append(t_stop)
                    length = subdata.size * BLOCK_SIZE
                    self._sigs_length.append(length)


# keys in
txt_header_keys = [
    ('AcqEntName', 'channel_names', None),  # used
    ('FileType', '', None),
    ('FileVersion', '', None),
    ('RecordSize', '', None),
    ('HardwareSubSystemName', '', None),
    ('HardwareSubSystemType', '', None),
    ('SamplingFrequency', 'sampling_rate', float),  # used
    ('ADMaxValue', '', None),
    ('ADBitVolts', 'bit_to_microVolt', None),  # used
    ('NumADChannels', '', None),
    ('ADChannel', 'channel_ids', None),  # used
    ('InputRange', '', None),
    ('InputInverted', 'input_inverted', bool),  # used
    ('DSPLowCutFilterEnabled', '', None),
    ('DspLowCutFrequency', '', None),
    ('DspLowCutNumTaps', '', None),
    ('DspLowCutFilterType', '', None),
    ('DSPHighCutFilterEnabled', '', None),
    ('DspHighCutFrequency', '', None),
    ('DspHighCutNumTaps', '', None),
    ('DspHighCutFilterType', '', None),
    ('DspDelayCompensation', '', None),
    ('DspFilterDelay_µs', '', None),
    ('DisabledSubChannels', '', None),
    ('WaveformLength', '', int),
    ('AlignmentPt', '', None),
    ('ThreshVal', '', None),
    ('MinRetriggerSamples', '', None),
    ('SpikeRetriggerTime', '', None),
    ('DualThresholding', '', None),
    ('Feature \w+ \d+', '', None),
    ('SessionUUID', '', None),
    ('FileUUID', '', None),
    ('CheetahRev', 'version', None),  # used  possibilty 1 for version
    ('ProbeName', '', None),
    ('OriginalFileName', '', None),
    ('TimeCreated', '', None),
    ('TimeClosed', '', None),
    ('ApplicationName Cheetah', 'version', None),  # used  possibilty 2 for version
    ('AcquisitionSystem', '', None),
    ('ReferenceChannel', '', None),
]


def read_txt_header(filename):
    """
    All file in neuralynx contains a 16kB hedaer in txt
    format.
    This function parse it to create info dict.
    This include datetime
    """
    with open(filename, 'rb') as f:
        txt_header = f.read(HEADER_SIZE)
    txt_header = txt_header.strip(b'\x00').decode('latin-1')

    # find keys
    info = OrderedDict()
    for k1, k2, type_ in txt_header_keys:
        pattern = '-(?P<name>' + k1 + ') (?P<value>[\S ]*)'
        matches = re.findall(pattern, txt_header)
        for match in matches:
            if k2 == '':
                name = match[0]
            else:
                name = k2
            value = match[1].rstrip(' ')
            if type_ is not None:
                value = type_(value)
            info[name] = value

    # if channel_ids or s not in info then the filename is used
    name = os.path.splitext(os.path.basename(filename))[0]

    # convert channel ids
    if 'channel_ids' in info:
        chid_entries = re.findall('\w+', info['channel_ids'])
        info['channel_ids'] = [int(c) for c in chid_entries]
    else:
        info['channel_ids'] = [name]

    # convert channel names
    if 'channel_names' in info:
        name_entries = re.findall('\w+', info['channel_names'])
        if len(name_entries) == 1:
            info['channel_names'] = name_entries * len(info['channel_ids'])
        assert len(info['channel_names']) == len(info['channel_ids']), \
            'Number of channel ids does not match channel names.'
    else:
        info['channel_names'] = [name] * len(info['channel_ids'])
    if 'version' in info:
        version = info['version'].replace('"', '')
        info['version'] = distutils.version.LooseVersion(version)

    # convert bit_to_microvolt
    if 'bit_to_microVolt' in info:
        btm_entries = re.findall('\S+', info['bit_to_microVolt'])
        if len(btm_entries) == 1:
            btm_entries = btm_entries * len(info['channel_ids'])
        info['bit_to_microVolt'] = [float(e) * 1e6 for e in btm_entries]
        assert len(info['bit_to_microVolt']) == len(info['channel_ids']), \
            'Number of channel ids does not match bit_to_microVolt conversion factors.'

    if 'InputRange' in info:
        ir_entries = re.findall('\w+', info['InputRange'])
        if len(ir_entries) == 1:
            info['InputRange'] = [int(ir_entries[0])] * len(chid_entries)
        else:
            info['InputRange'] = [int(e) for e in ir_entries]
        assert len(info['InputRange']) == len(chid_entries), \
            'Number of channel ids does not match input range values.'

    # filename and datetime
    if info['version'] <= distutils.version.LooseVersion('5.6.4'):
        datetime1_regex = '## Time Opened \(m/d/y\): (?P<date>\S+)  \(h:m:s\.ms\) (?P<time>\S+)'
        datetime2_regex = '## Time Closed \(m/d/y\): (?P<date>\S+)  \(h:m:s\.ms\) (?P<time>\S+)'
        filename_regex = '## File Name (?P<filename>\S+)'
        datetimeformat = '%m/%d/%Y %H:%M:%S.%f'
    else:
        datetime1_regex = '-TimeCreated (?P<date>\S+) (?P<time>\S+)'
        datetime2_regex = '-TimeClosed (?P<date>\S+) (?P<time>\S+)'
        filename_regex = '-OriginalFileName "?(?P<filename>\S+)"?'
        datetimeformat = '%Y/%m/%d %H:%M:%S'

    original_filename = re.search(filename_regex, txt_header).groupdict()['filename']

    dt1 = re.search(datetime1_regex, txt_header).groupdict()
    dt2 = re.search(datetime2_regex, txt_header).groupdict()

    info['recording_opened'] = datetime.datetime.strptime(
        dt1['date'] + ' ' + dt1['time'], datetimeformat)
    info['recording_closed'] = datetime.datetime.strptime(
        dt2['date'] + ' ' + dt2['time'], datetimeformat)

    return info


ncs_dtype = [('timestamp', 'uint64'), ('channel_id', 'uint32'), ('sample_rate', 'uint32'),
             ('nb_valid', 'uint32'), ('samples', 'int16', (BLOCK_SIZE,))]

nev_dtype = [
    ('reserved', '<i2'),
    ('system_id', '<i2'),
    ('data_size', '<i2'),
    ('timestamp', '<u8'),
    ('event_id', '<i2'),
    ('ttl_input', '<i2'),
    ('crc_check', '<i2'),
    ('dummy1', '<i2'),
    ('dummy2', '<i2'),
    ('extra', '<i4', (8,)),
    ('event_string', 'S128'),
]


def get_nse_or_ntt_dtype(info, ext):
    """
    For NSE and NTT the dtype depend on the header.

    """
    dtype = [('timestamp', 'uint64'), ('channel_id', 'uint32'), ('unit_id', 'uint32')]

    # count feature
    nb_feature = 0
    for k in info.keys():
        if k.startswith('Feature '):
            nb_feature += 1
    dtype += [('features', 'int32', (nb_feature,))]

    # count sample
    if ext == 'nse':
        nb_sample = info['WaveformLength']
        dtype += [('samples', 'int16', (nb_sample,))]
    elif ext == 'ntt':
        nb_sample = info['WaveformLength']
        nb_chan = 4  # check this if not tetrode
        dtype += [('samples', 'int16', (nb_sample, nb_chan))]

    return dtype
