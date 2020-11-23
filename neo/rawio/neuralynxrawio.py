"""
Class for reading data from Neuralynx files.
This IO supports NCS, NEV, NSE and NTT file formats.


NCS contains sampled signal for one channel
NEV contains events
NSE contains spikes and waveforms for mono electrodes
NTT contains spikes and waveforms for tetrodes


NCS can contains gaps that can be detected in irregularity
in timestamps of data blocks. Each gap leads to one new segment.
Some NCS files may need to be read entirely to detect those gaps which can be slow.


Author: Julia Sprenger, Carlos Canova, Samuel Garcia
"""
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

    def __init__(self, dirname='', keep_original_times=False, **kargs):
        """
        Parameters
        ----------
        dirname: str
            name of directory containing all files for dataset
        keep_original_times:
            if True, keep original start time as in files,
            otherwise set 0 of time to first time in dataset
        """
        self.dirname = dirname
        self.keep_original_times = keep_original_times
        BaseRawIO.__init__(self, **kargs)

    def _source_name(self):
        return self.dirname

    def _parse_header(self):

        sig_channels = []
        unit_channels = []
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

        for filename in sorted(os.listdir(self.dirname)):
            filename = os.path.join(self.dirname, filename)

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
            info = NlxHeader.build_for_file(filename)
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
                    group_id = 0
                    sig_channels.append((chan_name, chan_id, info['sampling_rate'],
                                         'int16', units, gain, offset, group_id))
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

                    if (os.path.getsize(filename) <= NlxHeader.HEADER_SIZE):
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
                        unit_channels.append(
                            (unit_name, '{}'.format(unit_id), wf_units, wf_gain,
                             wf_offset, wf_left_sweep, wf_sampling_rate))
                        unit_annotations.append(dict(file_origin=filename))

                elif ext == 'nev':
                    # an event channel
                    # each ('event_id',  'ttl_input') give a new event channel
                    self.nev_filenames[chan_id] = filename

                    if (os.path.getsize(filename) <= NlxHeader.HEADER_SIZE):
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

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # require all sampled signals, ncs files, to have same sampling rate
        if sig_channels.size > 0:
            sampling_rate = np.unique(sig_channels['sampling_rate'])
            assert sampling_rate.size == 1
            self._sigs_sampling_rate = sampling_rate[0]

        # set 2 attributes needed later for header in case there are no ncs files in dataset,
        #   e.g. Pegasus
        self._timestamp_limits = None
        self._nb_segment = 1

        # Read ncs files for gap detection and nb_segment computation.
        # :TODO: current algorithm depends on side-effect of read_ncs_files on
        #   self._sigs_memmap, self._sigs_t_start, self._sigs_t_stop,
        #   self._sigs_length, self._nb_segment, self._timestamp_limits
        ncsBlocks = self.read_ncs_files(self.ncs_filenames)

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

    # Accessors for segment times which are offset by appropriate global start time
    def _segment_t_start(self, block_index, seg_index):
        return self._seg_t_starts[seg_index] - self.global_t_start

    def _segment_t_stop(self, block_index, seg_index):
        return self._seg_t_stops[seg_index] - self.global_t_start

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        return self._sigs_length[seg_index]

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return self._sigs_t_start[seg_index] - self.global_t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
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

        block_start = i_start // BLOCK_SIZE
        block_stop = i_stop // BLOCK_SIZE + 1
        sl0 = i_start % 512
        sl1 = sl0 + (i_stop - i_start)

        if channel_indexes is None:
            channel_indexes = slice(None)

        channel_ids = self.header['signal_channels'][channel_indexes]['id']
        channel_names = self.header['signal_channels'][channel_indexes]['name']

        # create buffer for samples
        sigs_chunk = np.zeros((i_stop - i_start, len(channel_ids)), dtype='int16')

        for i, chan_uid in enumerate(zip(channel_names, channel_ids)):
            data = self._sigs_memmap[seg_index][chan_uid]
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

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        event_times = event_timestamps.astype(dtype)
        event_times /= 1e6
        event_times -= self.global_t_start
        return event_times

    def read_ncs_files(self, ncs_filenames):
        """
        Given a list of ncs files, return a dictionary of NcsBlocks indexed by channel uid.

        :TODO: Current algorithm has side effects on following attributes:
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
        # :TODO: Needs to account for gaps and start and end times potentially
        #    being different in different groups of channels. These groups typically
        #    correspond to the channels collected by a single ADC card.
        if len(ncs_filenames) == 0:
            return None

        good_delta = int(BLOCK_SIZE * 1e6 / self._sigs_sampling_rate)
        chan_uid0 = list(ncs_filenames.keys())[0]
        filename0 = ncs_filenames[chan_uid0]

        data0 = np.memmap(filename0, dtype=ncs_dtype, mode='r', offset=NlxHeader.HEADER_SIZE)

        gap_indexes = None
        lost_indexes = None

        if self.use_cache:
            gap_indexes = self._cache.get('gap_indexes')
            lost_indexes = self._cache.get('lost_indexes')

        # detect gaps on first file
        if (gap_indexes is None) or (lost_indexes is None):

            # this can be long!!!!
            timestamps0 = data0['timestamp']
            deltas0 = np.diff(timestamps0)

            # :TODO: This algorithm needs to account for older style files which had a rounded
            #   off sampling rate in the header.
            #
            # It should be that:
            # gap_indexes, = np.nonzero(deltas0!=good_delta)
            # but for a file I have found many deltas0==15999, 16000, 16001 (for sampling at 32000)
            # I guess this is a round problem
            # So this is the same with a tolerance of 1 or 2 ticks
            max_tolerance = 2
            mask = np.abs((deltas0 - good_delta).astype('int64')) > max_tolerance

            gap_indexes, = np.nonzero(mask)

            if self.use_cache:
                self.add_in_cache(gap_indexes=gap_indexes)

            # update for lost_indexes
            # Sometimes NLX writes a faulty block, but it then validates how much samples it wrote
            # the validation field is in delta0['nb_valid'], it should be equal to BLOCK_SIZE
            # :TODO: this algorithm ignores samples in partially filled blocks, which
            #   is not strictly necessary as all channels might have same partially filled
            #   blocks at the end.

            lost_indexes, = np.nonzero(data0['nb_valid'] < BLOCK_SIZE)

            if self.use_cache:
                self.add_in_cache(lost_indexes=lost_indexes)

        gap_candidates = np.unique([0]
                                   + [data0.size]
                                   + (gap_indexes + 1).tolist()
                                   + lost_indexes.tolist())  # linear

        gap_pairs = np.vstack([gap_candidates[:-1], gap_candidates[1:]]).T  # 2D (n_segments, 2)

        # construct proper gap ranges free of lost samples artifacts
        minimal_segment_length = 1  # in blocks
        goodpairs = np.diff(gap_pairs, 1).reshape(-1) > minimal_segment_length
        gap_pairs = gap_pairs[goodpairs]  # ensures a segment is at least a block wide

        self._nb_segment = len(gap_pairs)
        self._sigs_memmap = [{} for seg_index in range(self._nb_segment)]
        self._sigs_t_start = []
        self._sigs_t_stop = []
        self._sigs_length = []
        self._timestamp_limits = []

        # create segment with subdata block/t_start/t_stop/length
        for chan_uid, ncs_filename in self.ncs_filenames.items():

            data = np.memmap(ncs_filename, dtype=ncs_dtype, mode='r', offset=NlxHeader.HEADER_SIZE)
            assert data.size == data0.size, 'ncs files do not have the same data length'

            for seg_index, (i0, i1) in enumerate(gap_pairs):

                assert data[i0]['timestamp'] == data0[i0][
                    'timestamp'], 'ncs files do not have the same gaps'
                assert data[i1 - 1]['timestamp'] == data0[i1 - 1][
                    'timestamp'], 'ncs files do not have the same gaps'

                subdata = data[i0:i1]
                self._sigs_memmap[seg_index][chan_uid] = subdata

                if chan_uid == chan_uid0:
                    ts0 = subdata[0]['timestamp']
                    ts1 = subdata[-1]['timestamp'] +\
                        np.uint64(BLOCK_SIZE / self._sigs_sampling_rate * 1e6)
                    self._timestamp_limits.append((ts0, ts1))
                    t_start = ts0 / 1e6
                    self._sigs_t_start.append(t_start)
                    t_stop = ts1 / 1e6
                    self._sigs_t_stop.append(t_stop)
                    length = subdata.size * BLOCK_SIZE
                    self._sigs_length.append(length)


class NcsBlocks():
    """
    Contains information regarding the blocks of records in an Ncs file.
    Factory methods perform parsing of this information from an Ncs file or
    confirmation that file agrees with block structure.
    """

    startBlocks = []
    endBlocks = []
    sampFreqUsed = 0
    microsPerSampUsed = 0


class NlxHeader(OrderedDict):
    """
    Representation of basic information in all 16 kbytes Neuralynx file headers,
    including dates opened and closed if given.
    """

    HEADER_SIZE = 2 ** 14  # Neuralynx files have a txt header of 16kB

    # helper function to interpret boolean keys
    def _to_bool(txt):
        if txt == 'True':
            return True
        elif txt == 'False':
            return False
        else:
            raise Exception('Can not convert %s to bool' % (txt))

    # keys that may be present in header which we parse
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
        ('InputInverted', 'input_inverted', _to_bool),  # used
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
        (r'Feature \w+ \d+', '', None),
        ('SessionUUID', '', None),
        ('FileUUID', '', None),
        ('CheetahRev', '', None),  # only for older versions of Cheetah
        ('ProbeName', '', None),
        ('OriginalFileName', '', None),
        ('TimeCreated', '', None),
        ('TimeClosed', '', None),
        ('ApplicationName', '', None),  # also include version number when present
        ('AcquisitionSystem', '', None),
        ('ReferenceChannel', '', None),
        ('NLX_Base_Class_Type', '', None)  # in version 4 and earlier versions of Cheetah
    ]

    # Filename and datetime may appear in header lines starting with # at
    # beginning of header or in later versions as a property. The exact format
    # used depends on the application name and its version as well as the
    # -FileVersion property.
    #
    # There are 3 styles understood by this code and the patterns used for parsing
    # the items within each are stored in a dictionary. Each dictionary is then
    # stored in main dictionary keyed by an abbreviation for the style.
    header_pattern_dicts = {
        # Cheetah before version 5 and BML
        'bv5': dict(
            datetime1_regex=r'## Time Opened: \(m/d/y\): (?P<date>\S+)'
                            r'  At Time: (?P<time>\S+)',
            filename_regex=r'## File Name: (?P<filename>\S+)',
            datetimeformat='%m/%d/%Y %H:%M:%S.%f'),
        # Cheetah version 5 before and including v 5.6.4
        'bv5.6.4': dict(
            datetime1_regex=r'## Time Opened \(m/d/y\): (?P<date>\S+)'
                            r'  \(h:m:s\.ms\) (?P<time>\S+)',
            datetime2_regex=r'## Time Closed \(m/d/y\): (?P<date>\S+)'
                            r'  \(h:m:s\.ms\) (?P<time>\S+)',
            filename_regex=r'## File Name (?P<filename>\S+)',
            datetimeformat='%m/%d/%Y %H:%M:%S.%f'),
        # Cheetah after v 5.6.4 and default for others such as Pegasus
        'def': dict(
            datetime1_regex=r'-TimeCreated (?P<date>\S+) (?P<time>\S+)',
            datetime2_regex=r'-TimeClosed (?P<date>\S+) (?P<time>\S+)',
            filename_regex=r'-OriginalFileName "?(?P<filename>\S+)"?',
            datetimeformat='%Y/%m/%d %H:%M:%S')
    }

    def build_for_file(filename):
        """
        Factory function to build NlxHeader for a given file.
        """

        with open(filename, 'rb') as f:
            txt_header = f.read(NlxHeader.HEADER_SIZE)
        txt_header = txt_header.strip(b'\x00').decode('latin-1')

        # must start with 8 # characters
        assert txt_header.startswith("########"),\
            'Neuralynx files must start with 8 # characters.'

        # find keys
        info = NlxHeader()
        for k1, k2, type_ in NlxHeader.txt_header_keys:
            pattern = r'-(?P<name>' + k1 + r')\s+(?P<value>[\S ]*)'
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
            chid_entries = re.findall(r'\w+', info['channel_ids'])
            info['channel_ids'] = [int(c) for c in chid_entries]
        else:
            info['channel_ids'] = [name]

        # convert channel names
        if 'channel_names' in info:
            name_entries = re.findall(r'\w+', info['channel_names'])
            if len(name_entries) == 1:
                info['channel_names'] = name_entries * len(info['channel_ids'])
            assert len(info['channel_names']) == len(info['channel_ids']), \
                'Number of channel ids does not match channel names.'
        else:
            info['channel_names'] = [name] * len(info['channel_ids'])

        # version and application name
        # older Cheetah versions with CheetahRev property
        if 'CheetahRev' in info:
            assert 'ApplicationName' not in info
            info['ApplicationName'] = 'Cheetah'
            app_version = info['CheetahRev']
        # new file version 3.4 does not contain CheetahRev property, but ApplicationName instead
        elif 'ApplicationName' in info:
            pattern = r'(\S*) "([\S ]*)"'
            match = re.findall(pattern, info['ApplicationName'])
            assert len(match) == 1, 'impossible to find application name and version'
            info['ApplicationName'], app_version = match[0]
        # BML Ncs file writing contained neither property, assume BML version 2
        else:
            info['ApplicationName'] = 'BML'
            app_version = "2.0"

        info['ApplicationVersion'] = distutils.version.LooseVersion(app_version)

        # convert bit_to_microvolt
        if 'bit_to_microVolt' in info:
            btm_entries = re.findall(r'\S+', info['bit_to_microVolt'])
            if len(btm_entries) == 1:
                btm_entries = btm_entries * len(info['channel_ids'])
            info['bit_to_microVolt'] = [float(e) * 1e6 for e in btm_entries]
            assert len(info['bit_to_microVolt']) == len(info['channel_ids']), \
                'Number of channel ids does not match bit_to_microVolt conversion factors.'

        if 'InputRange' in info:
            ir_entries = re.findall(r'\w+', info['InputRange'])
            if len(ir_entries) == 1:
                info['InputRange'] = [int(ir_entries[0])] * len(chid_entries)
            else:
                info['InputRange'] = [int(e) for e in ir_entries]
            assert len(info['InputRange']) == len(chid_entries), \
                'Number of channel ids does not match input range values.'

        # Filename and datetime depend on app name, app version, and -FileVersion
        an = info['ApplicationName']
        if an == 'Cheetah':
            av = info['ApplicationVersion']
            if av < '5':
                hpd = NlxHeader.header_pattern_dicts['bv5']
            elif av <= '5.6.4':
                hpd = NlxHeader.header_pattern_dicts['bv5.6.4']
            else:
                hpd = NlxHeader.header_pattern_dicts['def']
        elif an == 'BML':
            hpd = NlxHeader.header_pattern_dicts['bv5']
        else:
            hpd = NlxHeader.header_pattern_dicts['def']

        original_filename = re.search(hpd['filename_regex'], txt_header).groupdict()['filename']

        # opening time
        dt1 = re.search(hpd['datetime1_regex'], txt_header).groupdict()
        info['recording_opened'] = datetime.datetime.strptime(
            dt1['date'] + ' ' + dt1['time'], hpd['datetimeformat'])

        # close time, if available
        if 'datetime2_regex' in hpd:
            dt2 = re.search(hpd['datetime2_regex'], txt_header).groupdict()
            info['recording_closed'] = datetime.datetime.strptime(
                dt2['date'] + ' ' + dt2['time'], hpd['datetimeformat'])

        return info

    def type_of_recording(self):
        """
        Determines type of recording in Ncs file with this header.

        RETURN:
            one of 'PRE4','BML','DIGITALLYNX','DIGITALLYNXSX','UNKNOWN'
        """

        if 'NLX_Base_Class_Type' in self:

            # older style standard neuralynx acquisition with rounded sampling frequency
            if self['NLX_Base_Class_Type'] == 'CscAcqEnt':
                return 'PRE4'

            # BML style with fractional frequency and microsPerSamp
            elif self['NLX_Base_Class_Type'] == 'BmlAcq':
                return 'BML'

            else:
                return 'UNKNOWN'

        elif 'HardwareSubSystemType' in self:

            # DigitalLynx
            if self['HardwareSubSystemType'] == 'DigitalLynx':
                return 'DIGITALLYNX'

            # DigitalLynxSX
            elif self['HardwareSubSystemType'] == 'DigitalLynxSX':
                return 'DIGITALLYNXSX'

        elif 'FileType' in self:

            if self['FileVersion'] in ['3.3', '3.4']:
                return self['AcquisitionSystem'].split()[1].upper()

            else:
                return 'UNKNOWN'

        else:
            return 'UNKNOWN'


class NcsHeader():
    """
    Representation of information in Ncs file headers, including exact
    recording type.
    """


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
