"""
Class for reading data from Neuralynx files.
This IO supports NCS, NEV, NSE and NTT file formats.


NCS contains the sampled signal for one channel
NEV contains events
NSE contains spikes and waveforms for mono electrodes
NTT contains spikes and waveforms for tetrodes

NCS files can contains gaps that can be detected in irregularity
in timestamps of data records. Each gap leads to one new segment being defined.
Some NCS files may need to be read entirely to detect those gaps, which can be slow.

Author: Julia Sprenger, Carlos Canova, Samuel Garcia, Peter N. Steinmetz.
"""


from ..baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)

import numpy as np
import os
from collections import (namedtuple, OrderedDict)

from neo.rawio.neuralynxrawio.ncssections import (NcsSection, NcsSectionsFactory)
from neo.rawio.neuralynxrawio.nlxheader import NlxHeader


class NeuralynxRawIO(BaseRawIO):
    """"
    Class for reading datasets recorded by Neuralynx.

    This version only works with rawmode of one-dir for a single directory.

    Examples:
        >>> reader = NeuralynxRawIO(dirname='Cheetah_v5.5.1/original_data')
        >>> reader.parse_header()

            Inspect all files in the directory.

        >>> print(reader)

            Display all information about signal channels, units, segment size....
    """
    extensions = ['nse', 'ncs', 'nev', 'ntt']
    rawmode = 'one-dir'

    _ncs_dtype = [('timestamp', 'uint64'), ('channel_id', 'uint32'), ('sample_rate', 'uint32'),
                  ('nb_valid', 'uint32'), ('samples', 'int16', (NcsSection._RECORD_SIZE))]

    def __init__(self, dirname='', filename='', keep_original_times=False, **kargs):
        """
        Initialize io for either a directory of Ncs files or a single Ncs file.

        Parameters
        ----------
        dirname: str
            name of directory containing all files for dataset. If provided, filename is
            ignored.
        filename: str
            name of a single ncs, nse, nev, or ntt file to include in dataset. If used,
            dirname must not be provided.
        keep_original_times:
            if True, keep original start time as in files,
            otherwise set 0 of time to first time in dataset
        """
        if dirname != '':
            self.dirname = dirname
            self.rawmode = 'one-dir'
        elif filename != '':
            self.filename = filename
            self.rawmode = 'one-file'
        else:
            raise ValueError("One of dirname or filename must be provided.")

        self.keep_original_times = keep_original_times
        BaseRawIO.__init__(self, **kargs)

    def _source_name(self):
        if self.rawmode == 'one-file':
            return self.filename
        else:
            return self.dirname

    def _parse_header(self):

        stream_channels = []
        signal_channels = []
        spike_channels = []
        event_channels = []

        self.ncs_filenames = OrderedDict()  # (chan_name, chan_id): filename
        self.nse_ntt_filenames = OrderedDict()  # (chan_name, chan_id): filename
        self.nev_filenames = OrderedDict()  # chan_id: filename

        self._nev_memmap = {}
        self._spike_memmap = {}
        self.internal_unit_ids = []  # channel_index > ((channel_name, channel_id), unit_id)
        self.internal_event_ids = []
        self._empty_ncs = []  # this list contains filenames of empty files
        self._empty_nev = []
        self._empty_nse_ntt = []

        # Explore the directory looking for ncs, nev, nse and ntt
        # and construct channels headers.
        signal_annotations = []
        unit_annotations = []
        event_annotations = []

        if self.rawmode == 'one-dir':
            filenames = sorted(os.listdir(self.dirname))
            dirname = self.dirname
        else:
            dirname, fname = os.path.split(self.filename)
            filenames = [fname]

        for filename in filenames:
            filename = os.path.join(dirname, filename)

            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # remove dot
            ext = ext.lower()  # make lower case for comparisons
            if ext not in self.extensions:
                continue

            # Skip Ncs files with only header. Other empty file types
            # will have an empty dataset constructed later.
            if (os.path.getsize(filename) <= NlxHeader.HEADER_SIZE) and ext in ['ncs']:
                self._empty_ncs.append(filename)
                continue

            # All file have more or less the same header structure
            info = NlxHeader(filename)
            chan_names = info['channel_names']
            chan_ids = info['channel_ids']

            for idx, chan_id in enumerate(chan_ids):
                chan_name = chan_names[idx]

                chan_uid = (chan_name, chan_id)
                if ext == 'ncs':
                    # a sampled signal channel
                    units = 'uV'
                    gain = info['bit_to_microVolt'][idx]
                    if info.get('input_inverted', False):
                        gain *= -1
                    offset = 0.
                    stream_id = 0
                    signal_channels.append((chan_name, str(chan_id), info['sampling_rate'],
                                         'int16', units, gain, offset, stream_id))
                    self.ncs_filenames[chan_uid] = filename
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
                    # nse and ntt are pretty similar except for the waveform shape.
                    # A file can contain several unit_id (so several unit channel).
                    assert chan_id not in self.nse_ntt_filenames, \
                        'Several nse or ntt files have the same unit_id!!!'
                    self.nse_ntt_filenames[chan_uid] = filename

                    dtype = get_nse_or_ntt_dtype(info, ext)

                    if os.path.getsize(filename) <= NlxHeader.HEADER_SIZE:
                        self._empty_nse_ntt.append(filename)
                        data = np.zeros((0,), dtype=dtype)
                    else:
                        data = np.memmap(filename, dtype=dtype, mode='r',
                                         offset=NlxHeader.HEADER_SIZE)

                    self._spike_memmap[chan_uid] = data

                    unit_ids = np.unique(data['unit_id'])
                    for unit_id in unit_ids:
                        # a spike channel for each (chan_id, unit_id)
                        self.internal_unit_ids.append((chan_uid, unit_id))

                        unit_name = "ch{}#{}#{}".format(chan_name, chan_id, unit_id)
                        unit_id = '{}'.format(unit_id)
                        wf_units = 'uV'
                        wf_gain = info['bit_to_microVolt'][idx]
                        if info.get('input_inverted', False):
                            wf_gain *= -1
                        wf_offset = 0.
                        wf_left_sweep = -1  # NOT KNOWN
                        wf_sampling_rate = info['sampling_rate']
                        spike_channels.append(
                            (unit_name, '{}'.format(unit_id), wf_units, wf_gain,
                             wf_offset, wf_left_sweep, wf_sampling_rate))
                        unit_annotations.append(dict(file_origin=filename))

                elif ext == 'nev':
                    # an event channel
                    # each ('event_id',  'ttl_input') give a new event channel
                    self.nev_filenames[chan_id] = filename

                    if os.path.getsize(filename) <= NlxHeader.HEADER_SIZE:
                        self._empty_nev.append(filename)
                        data = np.zeros((0,), dtype=nev_dtype)
                        internal_ids = []
                    else:
                        data = np.memmap(filename, dtype=nev_dtype, mode='r',
                                         offset=NlxHeader.HEADER_SIZE)
                        internal_ids = np.unique(data[['event_id', 'ttl_input']]).tolist()
                    for internal_event_id in internal_ids:
                        if internal_event_id not in self.internal_event_ids:
                            event_id, ttl_input = internal_event_id
                            name = '{} event_id={} ttl={}'.format(
                                chan_name, event_id, ttl_input)
                            event_channels.append((name, chan_id, 'event'))
                            self.internal_event_ids.append(internal_event_id)

                    self._nev_memmap[chan_id] = data

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # require all sampled signals, ncs files, to have same sampling rate
        if signal_channels.size > 0:
            sampling_rate = np.unique(signal_channels['sampling_rate'])
            assert sampling_rate.size == 1
            self._sigs_sampling_rate = sampling_rate[0]
            signal_streams = [('signals', '0')]
        else:
            signal_streams = []
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        # set 2 attributes needed later for header in case there are no ncs files in dataset,
        #   e.g. Pegasus
        self._timestamp_limits = None
        self._nb_segment = 1

        # Read ncs files for gap detection and nb_segment computation.
        self._sigs_memmaps, ncsSegTimestampLimits = self.scan_ncs_files(self.ncs_filenames)
        if ncsSegTimestampLimits:
            self._ncs_seg_timestamp_limits = ncsSegTimestampLimits  # save copy
            self._nb_segment = ncsSegTimestampLimits.nb_segment
            self._sigs_length = ncsSegTimestampLimits.length.copy()
            self._timestamp_limits = ncsSegTimestampLimits.timestamp_limits.copy()
            self._sigs_t_start = ncsSegTimestampLimits.t_start.copy()
            self._sigs_t_stop = ncsSegTimestampLimits.t_stop.copy()

        # Determine timestamp limits in nev, nse file by scanning them.
        ts0, ts1 = None, None
        for _data_memmap in (self._spike_memmap, self._nev_memmap):
            for _, data in _data_memmap.items():
                ts = data['timestamp']
                if ts.size == 0:
                    continue
                if ts0 is None:
                    ts0 = ts[0]
                    ts1 = ts[-1]
                ts0 = min(ts0, ts[0])
                ts1 = max(ts1, ts[-1])

        # decide on segment and global start and stop times based on files available
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

        if self.keep_original_times:
            self.global_t_stop = self.global_t_stop - self.global_t_start
            self.global_t_start = 0

        # fill header dictionary
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [self._nb_segment]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        # Annotations
        self._generate_minimal_annotations()
        bl_annotations = self.raw_annotations['blocks'][0]

        for seg_index in range(self._nb_segment):
            seg_annotations = bl_annotations['segments'][seg_index]

            for c in range(signal_streams.size):
                # one or no signal stream
                sig_ann = seg_annotations['signals'][c]
                # handle array annotations
                for key in signal_annotations[0].keys():
                    values = []
                    for c in range(signal_channels.size):
                        value = signal_annotations[0][key]
                        values.append(value)
                    values = np.array(values)
                    if values.ndim == 1:
                        # 'InputRange': is 2D and make bugs
                        sig_ann['__array_annotations__'][key] = values

            for c in range(spike_channels.size):
                unit_ann = seg_annotations['spikes'][c]
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

    # Accessors for segment times which are offset by appropriate global start time
    def _segment_t_start(self, block_index, seg_index):
        return self._seg_t_starts[seg_index] - self.global_t_start

    def _segment_t_stop(self, block_index, seg_index):
        return self._seg_t_stops[seg_index] - self.global_t_start

    def _get_signal_size(self, block_index, seg_index, stream_index):
        return self._sigs_length[seg_index]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return self._sigs_t_start[seg_index] - self.global_t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        """
        Retrieve chunk of analog signal, a chunk being a set of contiguous samples.

        PARAMETERS
        ----------
        block_index:
            index of block in dataset, ignored as only 1 block in this implementation
        seg_index:
            index of segment to use
        i_start:
            sample index of first sample within segment to retrieve
        i_stop:
            sample index of last sample within segment to retrieve
        channel_indexes:
            list of channel indices to return data for

        RETURNS
        -------
            array of samples, with each requested channel in a column
        """
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._sigs_length[seg_index]

        block_start = i_start // NcsSection._RECORD_SIZE
        block_stop = i_stop // NcsSection._RECORD_SIZE + 1
        sl0 = i_start % 512
        sl1 = sl0 + (i_stop - i_start)

        if channel_indexes is None:
            channel_indexes = slice(None)

        channel_ids = self.header['signal_channels'][channel_indexes]['id'].astype(int)
        channel_names = self.header['signal_channels'][channel_indexes]['name']

        # create buffer for samples
        sigs_chunk = np.zeros((i_stop - i_start, len(channel_ids)), dtype='int16')

        for i, chan_uid in enumerate(zip(channel_names, channel_ids)):
            data = self._sigs_memmaps[seg_index][chan_uid]
            sub = data[block_start:block_stop]
            sigs_chunk[:, i] = sub['samples'].flatten()[sl0:sl1]

        return sigs_chunk

    def _spike_count(self, block_index, seg_index, unit_index):
        chan_uid, unit_id = self.internal_unit_ids[unit_index]
        data = self._spike_memmap[chan_uid]
        ts = data['timestamp']

        ts0, ts1 = self._timestamp_limits[seg_index]

        # only count spikes inside the timestamp limits, inclusive, and for the specified unit
        keep = (ts >= ts0) & (ts <= ts1) & (unit_id == data['unit_id'])
        nb_spike = int(data[keep].size)
        return nb_spike

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        chan_uid, unit_id = self.internal_unit_ids[unit_index]
        data = self._spike_memmap[chan_uid]
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
        chan_uid, unit_id = self.internal_unit_ids[unit_index]
        data = self._spike_memmap[chan_uid]
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

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        event_times = event_timestamps.astype(dtype)
        event_times /= 1e6
        event_times -= self.global_t_start
        return event_times

    def scan_ncs_files(self, ncs_filenames):
        """
        Given a list of ncs files, read their basic structure.

        PARAMETERS:
        ------
        ncs_filenames - list of ncs filenames to scan.

        RETURNS:
        ------
        memmaps
            [ {} for seg_index in range(self._nb_segment) ][chan_uid]
        seg_time_limits
            SegmentTimeLimits for sections in scanned Ncs files

        Files will be scanned to determine the sections of records. If file is a single
        section of records, this scan is brief, otherwise it will check each record which may
        take some time.
        """

        # :TODO: Needs to account for gaps and start and end times potentially
        #    being different in different groups of channels. These groups typically
        #    correspond to the channels collected by a single ADC card.
        if len(ncs_filenames) == 0:
            return None, None

        # Build dictionary of chan_uid to associated NcsSections, memmap and NlxHeaders. Only
        # construct new NcsSections when it is different from that for the preceding file.
        chanSectMap = dict()
        for chan_uid, ncs_filename in self.ncs_filenames.items():

            data = np.memmap(ncs_filename, dtype=self._ncs_dtype, mode='r',
                             offset=NlxHeader.HEADER_SIZE)
            nlxHeader = NlxHeader(ncs_filename)

            if not chanSectMap or (chanSectMap and
                    not NcsSectionsFactory._verifySectionsStructure(data,
                    lastNcsSections)):
                lastNcsSections = NcsSectionsFactory.build_for_ncs_file(data, nlxHeader)

            chanSectMap[chan_uid] = [lastNcsSections, nlxHeader, data]

        # Construct an inverse dictionary from NcsSections to list of associated chan_uids
        revSectMap = dict()
        for k, v in chanSectMap.items():
            revSectMap.setdefault(v[0], []).append(k)

        # If there is only one NcsSections structure in the set of ncs files, there should only
        # be one entry. Otherwise this is presently unsupported.
        if len(revSectMap) > 1:
            raise IOError('ncs files have {} different sections structures. Unsupported.'.format(
                len(revSectMap)))

        seg_time_limits = SegmentTimeLimits(nb_segment=len(lastNcsSections.sects),
                                            t_start=[], t_stop=[], length=[],
                                            timestamp_limits=[])
        memmaps = [{} for seg_index in range(seg_time_limits.nb_segment)]

        # create segment with subdata block/t_start/t_stop/length for each channel
        for i, fileEntry in enumerate(self.ncs_filenames.items()):
            chan_uid = fileEntry[0]
            data = chanSectMap[chan_uid][2]

            # create a memmap for each record section of the current file
            curSects = chanSectMap[chan_uid][0]
            for seg_index in range(len(curSects.sects)):

                curSect = curSects.sects[seg_index]
                subdata = data[curSect.startRec:(curSect.endRec + 1)]
                memmaps[seg_index][chan_uid] = subdata

                # create segment timestamp limits based on only NcsSections structure in use
                if i == 0:
                    numSampsLastSect = subdata[-1]['nb_valid']
                    ts0 = subdata[0]['timestamp']
                    ts1 = NcsSectionsFactory.calc_sample_time(curSects.sampFreqUsed,
                                                              subdata[-1]['timestamp'],
                                                              numSampsLastSect)
                    seg_time_limits.timestamp_limits.append((ts0, ts1))
                    t_start = ts0 / 1e6
                    seg_time_limits.t_start.append(t_start)
                    t_stop = ts1 / 1e6
                    seg_time_limits.t_stop.append(t_stop)
                    # :NOTE: This should really be the total of nb_valid in records, but this
                    #  allows the last record of a section to be shorter, the most common case.
                    #  Have never seen a section of records with not full records before the last.
                    length = (subdata.size - 1) * NcsSection._RECORD_SIZE + numSampsLastSect
                    seg_time_limits.length.append(length)

        return memmaps, seg_time_limits


# time limits for set of segments
SegmentTimeLimits = namedtuple("SegmentTimeLimits", ['nb_segment', 't_start', 't_stop', 'length',
                                                     'timestamp_limits'])


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
