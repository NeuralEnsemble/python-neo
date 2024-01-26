"""
Class for reading data from Neuralynx files.
This IO supports NCS, NEV, NSE and NTT file formats.


NCS contains the sampled signal for one channel
NEV contains events
NSE contains spikes and waveforms for mono electrodes
NTT contains spikes and waveforms for tetrodes

All Neuralynx files contain a 16 kilobyte text header followed by 0 or more fixed length records.
The format of the header has never been formally specified, however, the Neuralynx programs which
write them have followed a set of conventions which have varied over the years. Additionally,
other programs like Pegasus write files with somewhat varying headers. This variation requires
parsing to determine the exact version and type which is handled within this RawIO by the
NlxHeader class.

Ncs files contain a series of 1044 byte records, each of which contains 512 16 byte samples and
header information which includes a 64 bit timestamp in microseconds, a 16 bit channel number,
the sampling frequency in integral Hz, and the number of the 512 samples which are considered
valid samples (the remaining samples within the record are invalid). The Ncs file header usually
contains a specification of the sampling frequency, which may be rounded to an integral number
of Hz or may be fractional. The actual sampling frequency in terms of the underlying clock is
physically determined by the spacing of the timestamps between records.

These variations of header format and possible differences between the stated sampling frequency
and actual sampling frequency can create apparent time discrepancies in .Ncs files. Additionally,
the Neuralynx recording software can start and stop recording while continuing to write samples
to a single .Ncs file, which creates larger gaps in the time sequences of the samples.

This RawIO attempts to correct for these deviations where possible and present a single section of
contiguous samples with one sampling frequency, t_start, and length for each .Ncs file. These
sections are determined by the NcsSectionsFactory class. In the
event the gaps are larger, this RawIO only provides the samples from the first section as belonging
to one Segment.

If .Ncs files are loaded these determine the Segments of data to be loaded. Events and spiking data
outside of Segments defined by .Ncs files will be ignored. To access all time point data in a
single Segment load a session excluding .Ncs files.

Continuous data streams are ordered by descending sampling rate.

This RawIO presents only a single Block.

Author: Julia Sprenger, Carlos Canova, Samuel Garcia, Peter N. Steinmetz.
"""

from ..baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                         _spike_channel_dtype, _event_channel_dtype)
from operator import itemgetter
import numpy as np
import os
import pathlib
import copy
from collections import (namedtuple, OrderedDict)

from neo.rawio.neuralynxrawio.ncssections import (NcsSection, NcsSectionsFactory)
from neo.rawio.neuralynxrawio.nlxheader import NlxHeader


class NeuralynxRawIO(BaseRawIO):
    """
    Class for reading datasets recorded by Neuralynx.

    This version works with rawmode of one-dir for a single directory of files or one-file for
    a single file.

    Examples:
        >>> reader = NeuralynxRawIO(dirname='Cheetah_v5.5.1/original_data')
        >>> reader.parse_header()

            Inspect all files in the directory.

        >>> print(reader)

            Display all information about signal channels, units, segment size....
    """
    extensions = ['nse', 'ncs', 'nev', 'ntt', 'nvt', 'nrd']  # nvt and nrd are not yet supported
    rawmode = 'one-dir'

    _ncs_dtype = [('timestamp', 'uint64'), ('channel_id', 'uint32'), ('sample_rate', 'uint32'),
                  ('nb_valid', 'uint32'), ('samples', 'int16', (NcsSection._RECORD_SIZE))]

    def __init__(self, dirname='', filename='', exclude_filename=None, keep_original_times=False,
                 **kargs):
        """
        Initialize io for either a directory of Ncs files or a single Ncs file.

        Parameters
        ----------
        dirname: str
            name of directory containing all files for dataset. If provided, filename is
            ignored.
        filename: str
            name of a single ncs, nse, nev, or ntt file to include in dataset. Will be ignored,
            if dirname is provided.
        exclude_filename: str or list
            name of a single ncs, nse, nev or ntt file or list of such files. Expects plain
            filenames (without directory path).
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
        self.exclude_filename = exclude_filename
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

        self.file_headers = OrderedDict()  # filename: file header dict

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
            if not os.path.isfile(self.filename):
                raise ValueError(f'Provided Filename is not a file: '
                                 f'{self.filename}. If you want to provide a '
                                 f'directory use the `dirname` keyword')

            dirname, fname = os.path.split(self.filename)
            filenames = [fname]

        if not isinstance(self.exclude_filename, (list, set, np.ndarray)):
            self.exclude_filename = [self.exclude_filename]

        # remove files that were explicitly excluded
        if self.exclude_filename is not None:
            for excl_file in self.exclude_filename:
                if excl_file in filenames:
                    filenames.remove(excl_file)

        stream_props = {}  # {(sampling_rate, n_samples, t_start): {stream_id: [filenames]}

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
            self.file_headers[filename] = info
            chan_names = info['channel_names']
            chan_ids = info['channel_ids']

            for idx, chan_id in enumerate(chan_ids):
                chan_name = chan_names[idx]

                chan_uid = (chan_name, str(chan_id))
                if ext == 'ncs':
                    file_mmap = self._get_file_map(filename)
                    n_packets = copy.copy(file_mmap.shape[0])
                    if n_packets:
                        t_start = copy.copy(file_mmap[0][0])
                    else:  # empty file
                        t_start = 0
                    stream_prop = (info['sampling_rate'], n_packets, t_start)
                    if stream_prop not in stream_props:
                        stream_props[stream_prop] = {'stream_id': len(stream_props),
                                                     'filenames': [filename]}
                    else:
                        stream_props[stream_prop]['filenames'].append(filename)
                    stream_id = stream_props[stream_prop]['stream_id']

                    # a sampled signal channel
                    units = 'uV'
                    gain = info['bit_to_microVolt'][idx]
                    if info.get('input_inverted', False):
                        gain *= -1
                    offset = 0.
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

                    data = self._get_file_map(filename)
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
                        data = self._get_file_map(filename)
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

        if signal_channels.size > 0:
            # ordering streams according from high to low sampling rates
            stream_props = {k: stream_props[k] for k in sorted(stream_props, reverse=True)}
            names = [f'Stream (rate,#packet,t0): {sp}' for sp in stream_props]
            ids = [stream_prop['stream_id'] for stream_prop in stream_props.values()]
            signal_streams = list(zip(names, ids))
        else:
            signal_streams = []
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        # set 2 attributes needed later for header in case there are no ncs files in dataset,
        #   e.g. Pegasus
        self._timestamp_limits = None
        self._nb_segment = 1

        stream_infos = {}

        # Read ncs files of each stream for gap detection and nb_segment computation.
        for stream_id in np.unique(signal_channels['stream_id']):
            stream_channels = signal_channels[signal_channels['stream_id'] == stream_id]
            stream_chan_uids = zip(stream_channels['name'], stream_channels['id'])
            stream_filenames = [self.ncs_filenames[chuid] for chuid in stream_chan_uids]
            _sigs_memmaps, ncsSegTimestampLimits, section_structure = self.scan_stream_ncs_files(
                stream_filenames)

            stream_infos[stream_id] = {'segment_sig_memmaps': _sigs_memmaps,
                                       'ncs_segment_infos': ncsSegTimestampLimits,
                                       'section_structure': section_structure}

        # check if section structure across streams is compatible and merge infos
        ref_sec_structure = None
        for stream_id, stream_info in stream_infos.items():
            ref_stream_id = list(stream_infos.keys())[0]
            ref_sec_structure = stream_infos[ref_stream_id]['section_structure']

            sec_structure = stream_info['section_structure']

            # check if section structure of streams are compatible
            # using tolerance of one data packet (512 samples)
            tolerance = 512 / min(ref_sec_structure.sampFreqUsed,
                                  sec_structure.sampFreqUsed) * 1e6
            if not ref_sec_structure.is_equivalent(sec_structure, abs_tol=tolerance):
                ref_chan_ids = signal_channels[signal_channels['stream_id'] == ref_stream_id][
                    'name']
                chan_ids = signal_channels[signal_channels['stream_id'] == stream_id]['name']

                raise ValueError('Incompatible section structures across streams: '
                                 f'Stream id {ref_stream_id}:{ref_chan_ids} and '
                                 f'{stream_id}:{chan_ids}.')

        if ref_sec_structure is not None:
            self._nb_segment = len(ref_sec_structure.sects)
        else:
            # Use only a single segment if no ncs data is present
            self._nb_segment = 1

        def min_max_tuple(tuple1, tuple2):
            """Merge tuple by selecting min for first and max for 2nd entry"""
            mins, maxs = zip(tuple1, tuple2)
            result = (min(m for m in mins if m is not None), max(m for m in maxs if m is not None))
            return result

        # merge stream mmemmaps since streams are compatible
        self._sigs_memmaps = [{} for seg_idx in range(self._nb_segment)]
        # time limits of integer timestamps in ncs files
        self._timestamp_limits = [(None, None) for seg_idx in range(self._nb_segment)]
        # time limits physical times in ncs files
        self._signal_limits = [(None, None) for seg_idx in range(self._nb_segment)]
        for stream_id, stream_info in stream_infos.items():
            stream_mmaps = stream_info['segment_sig_memmaps']
            for seg_idx, signal_dict in enumerate(stream_mmaps):
                self._sigs_memmaps[seg_idx].update(signal_dict)

            ncs_segment_info = stream_info['ncs_segment_infos']
            for seg_idx, (t_start, t_stop) in enumerate(ncs_segment_info.timestamp_limits):
                self._timestamp_limits[seg_idx] = min_max_tuple(self._timestamp_limits[seg_idx],
                                                                (t_start, t_stop))

            for seg_idx in range(ncs_segment_info.nb_segment):
                t_start = ncs_segment_info.t_start[seg_idx]
                t_stop = ncs_segment_info.t_stop[seg_idx]
                self._signal_limits[seg_idx] = min_max_tuple(self._signal_limits[seg_idx],
                                                             (t_start, t_stop))

        # precompute signal lengths within segments
        self._sigs_length = []
        if self._sigs_memmaps:
            for seg_idx, sig_container in enumerate(self._sigs_memmaps):
                self._sigs_length.append({})
                for chan_uid, sig_infos in sig_container.items():
                    self._sigs_length[seg_idx][chan_uid] = int(sig_infos['nb_valid'].sum())

        # Determine timestamp limits in nev, nse, ntt files by scanning them.
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

        # rescaling for comparison with signal times
        if ts0 is not None:
            timestamps_start, timestamps_stop = ts0 / 1e6, ts1 / 1e6

        # decide on segment and global start and stop times based on files available
        if self._timestamp_limits is None:
            # case  NO ncs but HAVE nev or nse -> single segment covering all spikes & events
            self._timestamp_limits = [(ts0, ts1)]
            self._seg_t_starts = [timestamps_start]
            self._seg_t_stops = [timestamps_stop]
            self.global_t_start = timestamps_start
            self.global_t_stop = timestamps_stop
        elif ts0 is not None:
            # case  HAVE ncs AND HAVE nev or nse -> multi segments based on ncs segmentation
            # ignoring nev/nse/ntt time limits, loading only data within ncs segments
            global_events_limits = (timestamps_start, timestamps_stop)
            global_signal_limits = (self._signal_limits[0][0], self._signal_limits[-1][-1])
            self.global_t_start, self.global_t_stop = min_max_tuple(global_events_limits,
                                                                    global_signal_limits)
            self._seg_t_starts = [limits[0] for limits in self._signal_limits]
            self._seg_t_stops = [limits[1] for limits in self._signal_limits]
            self._seg_t_starts[0] = self.global_t_start
            self._seg_t_stops[-1] = self.global_t_stop

        else:
            # case HAVE ncs but  NO nev or nse ->
            self._seg_t_starts = [limits[0] for limits in self._signal_limits]
            self._seg_t_stops = [limits[1] for limits in self._signal_limits]
            self.global_t_start = self._signal_limits[0][0]
            self.global_t_stop = self._signal_limits[-1][-1]

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

            for stream_id in range(signal_streams.size):
                # one or no signal stream
                stream_ann = seg_annotations['signals'][stream_id]
                # handle array annotations
                for key in signal_annotations[0].keys():
                    values = []
                    # only collect values from channels belonging to current stream
                    for d in np.where(signal_channels['stream_id'] == f'{stream_id}')[0]:
                        value = signal_annotations[d][key]
                        values.append(value)
                    values = np.array(values)
                    if values.ndim == 1:
                        # 'InputRange': is 2D and make bugs
                        stream_ann['__array_annotations__'][key] = values

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

    def _get_file_map(self, filename):
        """
        Create memory maps when needed
        see also https://github.com/numpy/numpy/issues/19340
        """
        filename = pathlib.Path(filename)
        suffix = filename.suffix.lower()[1:]

        if suffix == 'ncs':
            return np.memmap(filename, dtype=self._ncs_dtype, mode='r',
                             offset=NlxHeader.HEADER_SIZE)

        elif suffix in ['nse', 'ntt']:
            info = NlxHeader(filename)
            dtype = get_nse_or_ntt_dtype(info, suffix)

            # return empty map if file does not contain data
            if os.path.getsize(filename) <= NlxHeader.HEADER_SIZE:
                self._empty_nse_ntt.append(filename)
                return np.zeros((0,), dtype=dtype)

            return np.memmap(filename, dtype=dtype, mode='r',
                             offset=NlxHeader.HEADER_SIZE)

        elif suffix == 'nev':
            return np.memmap(filename, dtype=nev_dtype, mode='r',
                             offset=NlxHeader.HEADER_SIZE)

        else:
            raise ValueError(f'Unknown file suffix {suffix}')

    # Accessors for segment times which are offset by appropriate global start time
    def _segment_t_start(self, block_index, seg_index):
        return self._seg_t_starts[seg_index] - self.global_t_start

    def _segment_t_stop(self, block_index, seg_index):
        return self._seg_t_stops[seg_index] - self.global_t_start

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self.header['signal_streams'][stream_index]['id']
        stream_mask = self.header['signal_channels']['stream_id'] == stream_id
        signals = self.header['signal_channels'][stream_mask]

        if len(signals):
            sig = signals[0]
            return self._sigs_length[seg_index][(sig['name'], sig['id'])]
        else:
            raise ValueError(f'No signals present for block {block_index}, segment {seg_index},'
                             f' stream {stream_index}')

    def _get_signal_t_start(self, block_index, seg_index, stream_index):

        stream_id = self.header['signal_streams'][stream_index]['id']
        stream_mask = self.header['signal_channels']['stream_id'] == stream_id

        # use first channel of stream as all channels in stream have a common t_start
        channel = self.header['signal_channels'][stream_mask][0]

        data = self._sigs_memmaps[seg_index][(channel['name'], channel['id'])]
        absolute_t_start = data['timestamp'][0]

        return absolute_t_start / 1e6 - self.global_t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        """
        Retrieve chunk of analog signal, a chunk being a set of contiguous samples.

        Parameters
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

        Returns
        -------
            array of samples, with each requested channel in a column
        """
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.get_signal_size(block_index=block_index, seg_index=seg_index,
                                          stream_index=stream_index)

        block_start = i_start // NcsSection._RECORD_SIZE
        block_stop = i_stop // NcsSection._RECORD_SIZE + 1
        sl0 = i_start % 512
        sl1 = sl0 + (i_stop - i_start)

        if channel_indexes is None:
            channel_indexes = slice(None)

        stream_id = self.header['signal_streams'][stream_index]['id']
        stream_mask = self.header['signal_channels']['stream_id'] == stream_id

        channel_ids = self.header['signal_channels'][stream_mask][channel_indexes]['id']
        channel_names = self.header['signal_channels'][stream_mask][channel_indexes]['name']

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

        ts0 = self.segment_t_start(block_index, seg_index)
        ts1 = self.segment_t_stop(block_index, seg_index)

        # rescale to integer sampling of time
        ts0 = int((ts0 + self.global_t_start) * 1e6)
        ts1 = int((ts1 + self.global_t_start) * 1e6)


        # only count spikes inside the timestamp limits, inclusive, and for the specified unit
        keep = (ts >= ts0) & (ts <= ts1) & (unit_id == data['unit_id'])
        nb_spike = int(data[keep].size)
        return nb_spike

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        """
        Extract timestamps within a Segment defined by ncs timestamps
        """
        chan_uid, unit_id = self.internal_unit_ids[unit_index]
        data = self._spike_memmap[chan_uid]
        ts = data['timestamp']

        ts0, ts1 = t_start, t_stop
        if ts0 is None:
            ts0 = self.segment_t_start(block_index, seg_index)
        if ts1 is None:
            ts1 = self.segment_t_stop(block_index, seg_index)

        # rescale to integer sampling of time
        ts0 = int((ts0 + self.global_t_start) * 1e6)
        ts1 = int((ts1 + self.global_t_start) * 1e6)

        keep = (ts >= ts0) & (ts <= ts1) & (unit_id == data['unit_id'])
        timestamps = ts[keep]
        return timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= 1e6
        spike_times -= self.global_t_start
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        chan_uid, unit_id = self.internal_unit_ids[unit_index]
        data = self._spike_memmap[chan_uid]
        ts = data['timestamp']

        ts0, ts1 = t_start, t_stop
        if ts0 is None:
            ts0 = self.segment_t_start(block_index, seg_index)
        if ts1 is None:
            ts1 = self.segment_t_stop(block_index, seg_index)

        # rescale to integer sampling of time
        ts0 = int((ts0 + self.global_t_start) * 1e6)
        ts1 = int((ts1 + self.global_t_start) * 1e6)

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

        ts0 = self.segment_t_start(block_index, seg_index)
        ts1 = self.segment_t_stop(block_index, seg_index)

        # rescale to integer sampling of time
        ts0 = int((ts0 + self.global_t_start) * 1e6)
        ts1 = int((ts1 + self.global_t_start) * 1e6)

        ts = data['timestamp']
        keep = (ts >= ts0) & (ts <= ts1) & (data['event_id'] == event_id) & \
               (data['ttl_input'] == ttl_input)
        nb_event = int(data[keep].size)
        return nb_event

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        event_id, ttl_input = self.internal_event_ids[event_channel_index]
        chan_id = self.header['event_channels'][event_channel_index]['id']
        data = self._nev_memmap[chan_id]

        ts0, ts1 = t_start, t_stop
        if ts0 is None:
            ts0 = self.segment_t_start(block_index, seg_index)
        if ts1 is None:
            ts1 = self.segment_t_stop(block_index, seg_index)

        # rescale to integer sampling of time
        ts0 = int((ts0 + self.global_t_start) * 1e6)
        ts1 = int((ts1 + self.global_t_start) * 1e6)

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

    def scan_stream_ncs_files(self, ncs_filenames):
        """
        Given a list of ncs files, read their basic structure.
        Ncs files have to have common sampling_rate, number of packets and t_start
        (be part of a single stream)

        Parameters
        ----------
        ncs_filenames: list
            List of ncs filenames to scan.

        Returns
        -------
        memmaps
            [ {} for seg_index in range(self._nb_segment) ][chan_uid]
        seg_time_limits
            SegmentTimeLimits for sections in scanned Ncs files
        section_structure
            Section structure common to the ncs files

        Files will be scanned to determine the sections of records. If file is a single
        section of records, this scan is brief, otherwise it will check each record which may
        take some time.
        """

        if len(ncs_filenames) == 0:
            return None, None, None

        # Build dictionary of chan_uid to associated NcsSections, memmap and NlxHeaders. Only
        # construct new NcsSections when it is different from that for the preceding file.
        chanSectMap = dict()
        sig_length = []
        for ncs_filename in ncs_filenames:

            data = self._get_file_map(ncs_filename)
            nlxHeader = NlxHeader(ncs_filename)

            verify_sec_struct = NcsSectionsFactory._verifySectionsStructure
            if not chanSectMap or (not verify_sec_struct(data, chan_ncs_sections)):
                chan_ncs_sections = NcsSectionsFactory.build_for_ncs_file(data, nlxHeader)

            # register file section structure for all contained channels
            for chan_uid in zip(nlxHeader['channel_names'],
                                np.asarray(nlxHeader['channel_ids'], dtype=str)):
                chanSectMap[chan_uid] = [chan_ncs_sections, nlxHeader, ncs_filename]

            del data

        # Construct an inverse dictionary from NcsSections to list of associated chan_uids
        revSectMap = dict()
        for k, v in chanSectMap.items():
            revSectMap.setdefault(v[0], []).append(k)

        # If there is only one NcsSections structure in the set of ncs files, there should only
        # be one entry. Otherwise this is presently unsupported.
        if len(revSectMap) > 1:
            raise IOError(f'ncs files have {len(revSectMap)} different sections '
                          f'structures. Unsupported configuration to be handled with in a single '
                          f'stream.')

        seg_time_limits = SegmentTimeLimits(nb_segment=len(chan_ncs_sections.sects),
                                            t_start=[], t_stop=[], length=[],
                                            timestamp_limits=[])
        memmaps = [{} for seg_index in range(seg_time_limits.nb_segment)]

        # create segment with subdata block/t_start/t_stop/length for each channel
        for i, chan_uid in enumerate(chanSectMap.keys()):
            data = self._get_file_map(chanSectMap[chan_uid][2])

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

        stream_section_structure = list(revSectMap.keys())[0]

        return memmaps, seg_time_limits, stream_section_structure


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
