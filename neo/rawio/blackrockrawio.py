"""
Module for reading data from files in the Blackrock in raw format.

This work is based on:
  * Chris Rodgers - first version
  * Michael Denker, Lyuba Zehl - second version
  * Samuel Garcia - third version
  * Lyuba Zehl, Michael Denker - fourth version
  * Samuel Garcia, Julia Srenger - fifth version

This IO supports reading only.
This IO is able to read:
  * the nev file which contains spikes
  * ns1, ns2, .., ns6 files that contain signals at different sampling rates

This IO can handle the following Blackrock file specifications:
  * 2.1
  * 2.2
  * 2.3

The neural data channels are 1 - 128.
The analog inputs are 129 - 144. (129 - 137 AC coupled, 138 - 144 DC coupled)

spike- and event-data; 30000 Hz in NEV file.
"ns1": "analog data: 500 Hz",
"ns2": "analog data: 1000 Hz",
"ns3": "analog data: 2000 Hz",
"ns4": "analog data: 10000 Hz",
"ns5": "analog data: 30000 Hz",
"ns6": "analog data: 30000 Hz (no digital filter)"


The possible file extensions of the Cerebus system and their content:
    ns1: contains analog data; sampled at 500 Hz (+ digital filters)
    ns2: contains analog data; sampled at 1000 Hz (+ digital filters)
    ns3: contains analog data; sampled at 2000 Hz (+ digital filters)
    ns4: contains analog data; sampled at 10000 Hz (+ digital filters)
    ns5: contains analog data; sampled at 30000 Hz (+ digital filters)
    ns6: contains analog data; sampled at 30000 Hz (no digital filters)
    nev: contains spike- and event-data; sampled at 30000 Hz
    sif: contains institution and patient info (XML)
    ccf: contains Cerebus configurations

TODO:
  * videosync events (file spec 2.3)
  * tracking events (file spec 2.3)
  * buttontrigger events (file spec 2.3)
  * config events (file spec 2.3)
  * check left sweep settings of Blackrock
  * check nsx offsets (file spec 2.1)
  * add info of nev ext header (NSASEXEX) to non-neural events
    (file spec 2.1 and 2.2)
  * read sif file information
  * read ccf file information
  * fix reading of periodic sampling events (non-neural event type)
    (file spec 2.1 and 2.2)
"""


import datetime
import os
import re
import warnings

import numpy as np
import quantities as pq

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)


class BlackrockRawIO(BaseRawIO):
    """
    Class for reading data in from a file set recorded by the Blackrock
    (Cerebus) recording system.

    Upon initialization, the class is linked to the available set of Blackrock
    files.

    Note: This routine will handle files according to specification 2.1, 2.2,
    and 2.3. Recording pauses that may occur in file specifications 2.2 and
    2.3 are automatically extracted and the data set is split into different
    segments.

    The Blackrock data format consists not of a single file, but a set of
    different files. This constructor associates itself with a set of files
    that constitute a common data set. By default, all files belonging to
    the file set have the same base name, but different extensions.
    However, by using the override parameters, individual filenames can
    be set.

    Args:
        filename (string):
            File name (without extension) of the set of Blackrock files to
            associate with. Any .nsX or .nev, .sif, or .ccf extensions are
            ignored when parsing this parameter.
        nsx_override (string):
            File name of the .nsX files (without extension). If None,
            filename is used.
            Default: None.
        nev_override (string):
            File name of the .nev file (without extension). If None,
            filename is used.
            Default: None.
        nsx_to_load (int, list, 'max', 'all' (=None)) default None:
            IDs of nsX file from which to load data, e.g., if set to
            5 only data from the ns5 file are loaded.
            If 'all', then all nsX will be loaded.
            Contrary to previsous version of the IO  (<0.7), nsx_to_load
            must be set at the init before parse_header().

    Examples:
        >>> reader = BlackrockRawIO(filename='FileSpec2.3001', nsx_to_load=5)
        >>> reader.parse_header()

            Inspect a set of file consisting of files FileSpec2.3001.ns5 and
            FileSpec2.3001.nev

        >>> print(reader)

            Display all informations about signal channels, units, segment size....
    """

    extensions = ['ns' + str(_) for _ in range(1, 7)]
    extensions.extend(['nev', ])  # 'sif', 'ccf' not yet supported
    rawmode = 'multi-file'

    def __init__(self, filename=None, nsx_override=None, nev_override=None,
                 nsx_to_load=None, verbose=False):
        """
        Initialize the BlackrockIO class.
        """
        BaseRawIO.__init__(self)

        self.filename = str(filename)

        # remove extension from base _filenames
        for ext in self.extensions:
            if self.filename.endswith(os.path.extsep + ext):
                self.filename = self.filename.replace(os.path.extsep + ext, '')

        self.nsx_to_load = nsx_to_load

        # remove extensions from overrides
        self._filenames = {}
        if nsx_override:
            self._filenames['nsx'] = re.sub(
                os.path.extsep + 'ns[1,2,3,4,5,6]$', '', nsx_override)
        else:
            self._filenames['nsx'] = self.filename
        if nev_override:
            self._filenames['nev'] = re.sub(
                os.path.extsep + 'nev$', '', nev_override)
        else:
            self._filenames['nev'] = self.filename

        # check which files are available
        self._avail_files = dict.fromkeys(self.extensions, False)
        self._avail_nsx = []
        for ext in self.extensions:
            if ext.startswith('ns'):
                file2check = ''.join(
                    [self._filenames['nsx'], os.path.extsep, ext])
            else:
                file2check = ''.join(
                    [self._filenames[ext], os.path.extsep, ext])

            if os.path.exists(file2check):
                self._avail_files[ext] = True
                if ext.startswith('ns'):
                    self._avail_nsx.append(int(ext[-1]))

        if not self._avail_files['nev'] and not self._avail_nsx:
            raise IOError("No Blackrock files found in specified path")

        # These dictionaries are used internally to map the file specification
        # revision of the nsx and nev files to one of the reading routines
        # NSX
        self.__nsx_header_reader = {
            '2.1': self.__read_nsx_header_variant_a,
            '2.2': self.__read_nsx_header_variant_b,
            '2.3': self.__read_nsx_header_variant_b}
        self.__nsx_dataheader_reader = {
            '2.1': self.__read_nsx_dataheader_variant_a,
            '2.2': self.__read_nsx_dataheader_variant_b,
            '2.3': self.__read_nsx_dataheader_variant_b}
        self.__nsx_data_reader = {
            '2.1': self.__read_nsx_data_variant_a,
            '2.2': self.__read_nsx_data_variant_b,
            '2.3': self.__read_nsx_data_variant_b}
        self.__nsx_params = {
            '2.1': self.__get_nsx_param_variant_a,
            '2.2': self.__get_nsx_param_variant_b,
            '2.3': self.__get_nsx_param_variant_b}
        # NEV
        self.__nev_header_reader = {
            '2.1': self.__read_nev_header_variant_a,
            '2.2': self.__read_nev_header_variant_b,
            '2.3': self.__read_nev_header_variant_c}
        self.__nev_data_reader = {
            '2.1': self.__read_nev_data_variant_a,
            '2.2': self.__read_nev_data_variant_a,
            '2.3': self.__read_nev_data_variant_b}
        self.__waveform_size = {
            '2.1': self.__get_waveform_size_variant_a,
            '2.2': self.__get_waveform_size_variant_a,
            '2.3': self.__get_waveform_size_variant_b}
        self.__channel_labels = {
            '2.1': self.__get_channel_labels_variant_a,
            '2.2': self.__get_channel_labels_variant_b,
            '2.3': self.__get_channel_labels_variant_b}
        self.__nonneural_evdicts = {
            '2.1': self.__get_nonneural_evdicts_variant_a,
            '2.2': self.__get_nonneural_evdicts_variant_a,
            '2.3': self.__get_nonneural_evdicts_variant_b}
        self.__comment_evdict = {
            '2.1': self.__get_comment_evdict_variant_a,
            '2.2': self.__get_comment_evdict_variant_a,
            '2.3': self.__get_comment_evdict_variant_a}

    def _parse_header(self):

        main_sampling_rate = 30000.

        event_channels = []
        spike_channels = []
        signal_streams = []
        signal_channels = []

        # Step1 NEV file
        if self._avail_files['nev']:
            # Load file spec and headers of available

            # read nev file specification
            self.__nev_spec = self.__extract_nev_file_spec()

            # read nev headers
            self.__nev_basic_header, self.__nev_ext_header = \
                self.__nev_header_reader[self.__nev_spec]()

            self.nev_data = self.__nev_data_reader[self.__nev_spec]()
            spikes, spike_segment_ids = self.nev_data['Spikes']

            # scan all channel to get number of Unit
            spike_channels = []
            self.internal_unit_ids = []  # pair of chan['packet_id'], spikes['unit_class_nb']
            for i in range(len(self.__nev_ext_header[b'NEUEVWAV'])):

                channel_id = self.__nev_ext_header[b'NEUEVWAV']['electrode_id'][i]

                chan_mask = (spikes['packet_id'] == channel_id)
                chan_spikes = spikes[chan_mask]
                all_unit_id = np.unique(chan_spikes['unit_class_nb'])
                for u, unit_id in enumerate(all_unit_id):
                    self.internal_unit_ids.append((channel_id, unit_id))
                    name = "ch{}#{}".format(channel_id, unit_id)
                    _id = "Unit {}".format(1000 * channel_id + unit_id)
                    wf_gain = self.__nev_params('digitization_factor')[channel_id] / 1000.
                    wf_offset = 0.
                    wf_units = 'uV'
                    # TODO: Double check if this is the correct assumption (10 samples)
                    # default value: threshold crossing after 10 samples of waveform
                    wf_left_sweep = 10
                    wf_sampling_rate = main_sampling_rate
                    spike_channels.append((name, _id, wf_units, wf_gain,
                                          wf_offset, wf_left_sweep, wf_sampling_rate))

            # scan events
            # NonNeural: serial and digital input
            events_data, event_segment_ids = self.nev_data['NonNeural']
            ev_dict = self.__nonneural_evdicts[self.__nev_spec](events_data)
            if 'Comments' in self.nev_data:
                comments_data, comments_segment_ids = self.nev_data['Comments']
                ev_dict.update(self.__comment_evdict[self.__nev_spec](comments_data))
            for ev_name in ev_dict:
                event_channels.append((ev_name, '', 'event'))
            # TODO: TrackingEvents
            # TODO: ButtonTrigger
            # TODO: VideoSync

        # Step2 NSX file
        # Load file spec and headers of available nsx files
        self.__nsx_spec = {}
        self.__nsx_basic_header = {}
        self.__nsx_ext_header = {}
        self.__nsx_data_header = {}

        for nsx_nb in self._avail_nsx:
            spec = self.__nsx_spec[nsx_nb] = self.__extract_nsx_file_spec(nsx_nb)
            # read nsx headers
            self.__nsx_basic_header[nsx_nb], self.__nsx_ext_header[nsx_nb] = \
                self.__nsx_header_reader[spec](nsx_nb)

            # Read nsx data header(s)
            # for nsxdef get_analogsignal_shape(self, block_index, seg_index):
            self.__nsx_data_header[nsx_nb] = self.__nsx_dataheader_reader[spec](nsx_nb)

        # nsx_to_load can be either int, list, 'max', all' (aka None)
        # here make a list only
        if self.nsx_to_load is None or self.nsx_to_load == 'all':
            self.nsx_to_load = list(self._avail_nsx)
        elif self.nsx_to_load == 'max':
            if len(self._avail_nsx):
                self.nsx_to_load = [max(self._avail_nsx)]
            else:
                self.nsx_to_load = []
        elif isinstance(self.nsx_to_load, int):
            self.nsx_to_load = [self.nsx_to_load]
        elif isinstance(self.nsx_to_load, list):
            pass
        else:
            raise(ValueError('nsx_to_load is wrong'))

        assert all(nsx_nb in self._avail_nsx for nsx_nb in self.nsx_to_load),\
               'nsx_to_load do not match available nsx list'

        # check that all files come from the same specification
        all_spec = [self.__nsx_spec[nsx_nb] for nsx_nb in self.nsx_to_load]
        if self._avail_files['nev']:
            all_spec.append(self.__nev_spec)
        assert all(all_spec[0] == spec for spec in all_spec), \
            "Files don't have the same internal version"

        if len(self.nsx_to_load) > 0 and \
                self.__nsx_spec[self.nsx_to_load[0]] == '2.1' and \
                not self._avail_files['nev']:
            pass
            # Because rescaling to volts requires information from nev file (dig_factor)
            # Remove if raw loading becomes possible
            # raise IOError("For loading Blackrock file version 2.1 .nev files are required!")

        # This requires nsX to be parsed already
        # Needs to be called when no nsX are available as well in order to warn the user
        if self._avail_files['nev']:
            for nsx_nb in self.nsx_to_load:
                self.__match_nsx_and_nev_segment_ids(nsx_nb)

        self.nsx_datas = {}
        self.sig_sampling_rates = {}
        if len(self.nsx_to_load) > 0:
            for nsx_nb in self.nsx_to_load:
                spec = self.__nsx_spec[nsx_nb]
                self.nsx_datas[nsx_nb] = self.__nsx_data_reader[spec](nsx_nb)

                sr = float(main_sampling_rate / self.__nsx_basic_header[nsx_nb]['period'])
                self.sig_sampling_rates[nsx_nb] = sr

                if spec in ['2.2', '2.3']:
                    ext_header = self.__nsx_ext_header[nsx_nb]
                elif spec == '2.1':
                    ext_header = []
                    keys = ['labels', 'units', 'min_analog_val',
                            'max_analog_val', 'min_digital_val', 'max_digital_val']
                    params = self.__nsx_params[spec](nsx_nb)
                    for i in range(len(params['labels'])):
                        d = {}
                        for key in keys:
                            d[key] = params[key][i]
                        ext_header.append(d)

                if len(ext_header) > 0:
                    signal_streams.append((f'nsx{nsx_nb}', str(nsx_nb)))
                for i, chan in enumerate(ext_header):
                    if spec in ['2.2', '2.3']:
                        ch_name = chan['electrode_label'].decode()
                        ch_id = str(chan['electrode_id'])
                        units = chan['units'].decode()
                    elif spec == '2.1':
                        ch_name = chan['labels']
                        ch_id = str(self.__nsx_ext_header[nsx_nb][i]['electrode_id'])
                        units = chan['units']
                    sig_dtype = 'int16'
                    # max_analog_val/min_analog_val/max_digital_val/min_analog_val are int16!!!!!
                    # dangarous situation so cast to float everyone
                    if np.isnan(float(chan['min_analog_val'])):
                        gain = 1
                        offset = 0
                    else:
                        gain = (float(chan['max_analog_val']) - float(chan['min_analog_val'])) / \
                               (float(chan['max_digital_val']) - float(chan['min_digital_val']))
                        offset = -float(chan['min_digital_val']) \
                            * gain + float(chan['min_analog_val'])
                    stream_id = str(nsx_nb)
                    signal_channels.append((ch_name, ch_id, sr, sig_dtype,
                                         units, gain, offset, stream_id))

            # check nb segment per nsx
            nb_segments_for_nsx = [len(self.nsx_datas[nsx_nb]) for nsx_nb in self.nsx_to_load]
            assert all(nb == nb_segments_for_nsx[0] for nb in nb_segments_for_nsx),\
                   'Segment nb not consistanent across nsX files'
            self._nb_segment = nb_segments_for_nsx[0]

            self.__delete_empty_segments()

            # t_start/t_stop for segment are given by nsx limits or nev limits
            self._sigs_t_starts = {nsx_nb: [] for nsx_nb in self.nsx_to_load}
            self._seg_t_starts, self._seg_t_stops = [], []
            for data_bl in range(self._nb_segment):
                t_stop = 0.
                for nsx_nb in self.nsx_to_load:
                    length = self.nsx_datas[nsx_nb][data_bl].shape[0]
                    if self.__nsx_data_header[nsx_nb] is None:
                        t_start = 0.
                    else:
                        t_start = self.__nsx_data_header[nsx_nb][data_bl]['timestamp'] / \
                            self.__nsx_basic_header[nsx_nb]['timestamp_resolution']
                    t_stop = max(t_stop, t_start + length / self.sig_sampling_rates[nsx_nb])
                    self._sigs_t_starts[nsx_nb].append(t_start)

                if self._avail_files['nev']:
                    max_nev_time = 0
                    for k, (data, ev_ids) in self.nev_data.items():
                        segment_mask = ev_ids == data_bl
                        if data[segment_mask].size > 0:
                            t = data[segment_mask][-1]['timestamp'] / self.__nev_basic_header[
                                'timestamp_resolution']
                            max_nev_time = max(max_nev_time, t)
                    if max_nev_time > t_stop:
                        t_stop = max_nev_time
                    min_nev_time = max_nev_time
                    for k, (data, ev_ids) in self.nev_data.items():
                        segment_mask = ev_ids == data_bl
                        if data[segment_mask].size > 0:
                            t = data[segment_mask][0]['timestamp'] / self.__nev_basic_header[
                                'timestamp_resolution']
                            min_nev_time = min(min_nev_time, t)
                    if min_nev_time < t_start:
                        t_start = min_nev_time
                self._seg_t_starts.append(t_start)
                self._seg_t_stops.append(float(t_stop))

        else:
            # When only nev is available, only segments that are documented in nev can be detected

            max_nev_times = {}
            min_nev_times = {}

            # Find maximal and minimal time for each nev segment
            for k, (data, ev_ids) in self.nev_data.items():
                for i in np.unique(ev_ids):
                    mask = [ev_ids == i]
                    curr_data = data[mask]
                    if curr_data.size > 0:
                        if max(curr_data['timestamp']) >= max_nev_times.get(i, 0):
                            max_nev_times[i] = max(curr_data['timestamp'])
                        if min(curr_data['timestamp']) <= min_nev_times.get(i,
                                                                            max_nev_times[i]):
                            min_nev_times[i] = min(curr_data['timestamp'])

            # Calculate t_start and t_stop for each segment in seconds
            resolution = self.__nev_basic_header['timestamp_resolution']
            self._seg_t_starts = [v / float(resolution) for k, v in sorted(min_nev_times.items())]
            self._seg_t_stops = [v / float(resolution) for k, v in sorted(max_nev_times.items())]
            self._nb_segment = len(self._seg_t_starts)
            self._sigs_t_starts = [None] * self._nb_segment

        # finalize header
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [self._nb_segment]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        rec_datetime = self.__nev_params('rec_datetime') if self._avail_files['nev'] else None

        # Put annotations at some places for compatibility
        # with previous BlackrockIO version
        self._generate_minimal_annotations()

        block_ann = self.raw_annotations['blocks'][0]
        block_ann['description'] = 'Block of data from Blackrock file set.'
        block_ann['file_origin'] = self.filename
        block_ann['name'] = "Blackrock Data Block"
        block_ann['rec_datetime'] = rec_datetime
        block_ann['avail_file_set'] = [k for k, v in self._avail_files.items() if v]
        block_ann['avail_nsx'] = self._avail_nsx
        block_ann['avail_nev'] = self._avail_files['nev']
        #  'sif' and 'ccf' files not yet supported
        # block_ann['avail_sif'] = self._avail_files['sif']
        # block_ann['avail_ccf'] = self._avail_files['ccf']
        block_ann['rec_pauses'] = False

        for seg_index in range(self._nb_segment):
            seg_ann = block_ann['segments'][seg_index]
            seg_ann['file_origin'] = self.filename
            seg_ann['name'] = "Segment {}".format(seg_index)
            seg_ann['description'] = "Segment containing data from t_start to t_stop"
            if seg_index == 0:
                # if more than 1 segment means pause
                # so datetime is valide only for seg_index=0
                seg_ann['rec_datetime'] = rec_datetime

            for c in range(signal_streams.size):
                sig_ann = seg_ann['signals'][c]
                stream_id = signal_streams['id'][c]
                nsx_nb = int(stream_id)
                sig_ann['description'] = f'AnalogSignal from  nsx{nsx_nb}'
                sig_ann['file_origin'] = self._filenames['nsx'] + '.ns' + str(nsx_nb)
                sig_ann['nsx'] = nsx_nb
                # handle signal array annotations from nsx header
                if self.__nsx_spec[nsx_nb] in ['2.2', '2.3'] and nsx_nb in self.__nsx_ext_header:
                    mask = signal_channels['stream_id'] == stream_id
                    channels = signal_channels[mask]
                    nsx_header = self.__nsx_ext_header[nsx_nb]
                    for key in ('physical_connector', 'connector_pin', 'hi_freq_corner',
                                'lo_freq_corner', 'hi_freq_order', 'lo_freq_order',
                                'hi_freq_type', 'lo_freq_type'):
                        values = []
                        for chan_id in channels['id']:
                            chan_id = int(chan_id)
                            idx = list(nsx_header['electrode_id']).index(chan_id)
                            values.append(nsx_header[key][idx])
                        values = np.array(values)
                        sig_ann['__array_annotations__'][key] = values

            for c in range(spike_channels.size):
                st_ann = seg_ann['spikes'][c]
                channel_id, unit_id = self.internal_unit_ids[c]
                unit_tag = {0: 'unclassified', 255: 'noise'}.get(unit_id, str(unit_id))
                st_ann['channel_id'] = channel_id
                st_ann['unit_id'] = unit_id
                st_ann['unit_tag'] = unit_tag
                st_ann['description'] = f'SpikeTrain channel_id: {channel_id}, unit_id: {unit_id}'
                st_ann['file_origin'] = self._filenames['nev'] + '.nev'

            if self._avail_files['nev']:
                ev_dict = self.__nonneural_evdicts[self.__nev_spec](events_data)
                if 'Comments' in self.nev_data:
                    ev_dict.update(self.__comment_evdict[self.__nev_spec](comments_data))
                    color_codes = ["#{:08X}".format(code) for code in comments_data['color']]
                    color_codes = np.array(color_codes, dtype='S9')
                for c in range(event_channels.size):
                    # Next line makes ev_ann a reference to seg_ann['events'][c]
                    ev_ann = seg_ann['events'][c]
                    name = event_channels['name'][c]
                    ev_ann['description'] = ev_dict[name]['desc']
                    ev_ann['file_origin'] = self._filenames['nev'] + '.nev'
                    if name == 'comments':
                        ev_ann['color_codes'] = color_codes

    def _source_name(self):
        return self.filename

    def _segment_t_start(self, block_index, seg_index):
        return self._seg_t_starts[seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        return self._seg_t_stops[seg_index]

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self.header['signal_streams'][stream_index]['id']
        nsx_nb = int(stream_id)
        memmap_data = self.nsx_datas[nsx_nb][seg_index]
        return memmap_data.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        stream_id = self.header['signal_streams'][stream_index]['id']
        nsx_nb = int(stream_id)
        return self._sigs_t_starts[nsx_nb][seg_index]

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        stream_id = self.header['signal_streams'][stream_index]['id']
        nsx_nb = int(stream_id)
        memmap_data = self.nsx_datas[nsx_nb][seg_index]
        if channel_indexes is None:
            channel_indexes = slice(None)
        sig_chunk = memmap_data[i_start:i_stop, channel_indexes]
        return sig_chunk

    def _spike_count(self, block_index, seg_index, unit_index):
        channel_id, unit_id = self.internal_unit_ids[unit_index]

        all_spikes = self.nev_data['Spikes'][0]
        mask = (all_spikes['packet_id'] == channel_id) & (all_spikes['unit_class_nb'] == unit_id)
        if self._nb_segment == 1:
            # very fast
            nb = int(np.sum(mask))
        else:
            # must clip in time time range
            timestamp = all_spikes[mask]['timestamp']
            sl = self._get_timestamp_slice(timestamp, seg_index, None, None)
            timestamp = timestamp[sl]
            nb = timestamp.size
        return nb

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        channel_id, unit_id = self.internal_unit_ids[unit_index]

        all_spikes, event_segment_ids = self.nev_data['Spikes']

        # select by channel_id and unit_id
        mask = ((all_spikes['packet_id'] == channel_id) & (all_spikes['unit_class_nb'] == unit_id)
                & (event_segment_ids == seg_index))
        unit_spikes = all_spikes[mask]

        timestamp = unit_spikes['timestamp']
        sl = self._get_timestamp_slice(timestamp, seg_index, t_start, t_stop)
        timestamp = timestamp[sl]

        return timestamp

    def _get_timestamp_slice(self, timestamp, seg_index, t_start, t_stop):
        if self._nb_segment > 1:
            # we must clip event in seg time limits
            if t_start is None:
                t_start = self._seg_t_starts[seg_index]
            if t_stop is None:
                t_stop = self._seg_t_stops[seg_index]

        if t_start is None:
            ind_start = None
        else:
            ts = np.math.ceil(t_start * self.__nev_basic_header['timestamp_resolution'])
            ind_start = np.searchsorted(timestamp, ts)

        if t_stop is None:
            ind_stop = None
        else:
            ts = int(t_stop * self.__nev_basic_header['timestamp_resolution'])
            ind_stop = np.searchsorted(timestamp, ts)  # +1

        sl = slice(ind_start, ind_stop)
        return sl

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self.__nev_basic_header['timestamp_resolution']
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        channel_id, unit_id = self.internal_unit_ids[unit_index]
        all_spikes, event_segment_ids = self.nev_data['Spikes']

        mask = ((all_spikes['packet_id'] == channel_id) & (all_spikes['unit_class_nb'] == unit_id)
                & (event_segment_ids == seg_index))
        unit_spikes = all_spikes[mask]

        wf_dtype = self.__nev_params('waveform_dtypes')[channel_id]
        wf_size = self.__nev_params('waveform_size')[channel_id]

        waveforms = unit_spikes['waveform'].flatten().view(wf_dtype)
        waveforms = waveforms.reshape(int(unit_spikes.size), 1, int(wf_size))

        timestamp = unit_spikes['timestamp']
        sl = self._get_timestamp_slice(timestamp, seg_index, t_start, t_stop)
        waveforms = waveforms[sl]

        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        name = self.header['event_channels']['name'][event_channel_index]
        if name == 'comments':
            events_data, event_segment_ids = self.nev_data['Comments']
            ev_dict = self.__comment_evdict[self.__nev_spec](events_data)[name]
        else:
            events_data, event_segment_ids = self.nev_data['NonNeural']
            ev_dict = self.__nonneural_evdicts[self.__nev_spec](events_data)[name]
        mask = ev_dict['mask'] & (event_segment_ids == seg_index)
        if self._nb_segment == 1:
            # very fast
            nb = int(np.sum(mask))
        else:
            # must clip in time time range
            timestamp = events_data[ev_dict['mask']]['timestamp']
            sl = self._get_timestamp_slice(timestamp, seg_index, None, None)
            timestamp = timestamp[sl]
            nb = timestamp.size
        return nb

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        name = self.header['event_channels']['name'][event_channel_index]
        if name == 'comments':
            events_data, event_segment_ids = self.nev_data['Comments']
            ev_dict = self.__comment_evdict[self.__nev_spec](events_data)[name]
            # If immediate decoding is desired:
            encoding = {0: 'latin_1', 1: 'utf_16', 255: 'latin_1'}
            labels = [data[ev_dict['field']].decode(
                encoding[data['char_set']]) for data in events_data]
            labels = np.array(labels, dtype='U')
        else:
            events_data, event_segment_ids = self.nev_data['NonNeural']
            ev_dict = self.__nonneural_evdicts[self.__nev_spec](events_data)[name]
            labels = events_data[ev_dict['field']].astype('U')

        mask = ev_dict['mask'] & (event_segment_ids == seg_index)
        timestamp = events_data[mask]['timestamp']
        labels = labels[mask]

        # time clip
        sl = self._get_timestamp_slice(timestamp, seg_index, t_start, t_stop)
        timestamp = timestamp[sl]
        labels = labels[sl]
        durations = None

        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        ev_times = event_timestamps.astype(dtype)
        ev_times /= self.__nev_basic_header['timestamp_resolution']
        return ev_times

    ###################################################
    ###################################################

    # Above here code from Lyuba Zehl, Michael Denker
    # coming from previous BlackrockIO

    def __extract_nsx_file_spec(self, nsx_nb):
        """
        Extract file specification from an .nsx file.
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        # Header structure of files specification 2.2 and higher. For files 2.1
        # and lower, the entries ver_major and ver_minor are not supported.
        dt0 = [
            ('file_id', 'S8'),
            ('ver_major', 'uint8'),
            ('ver_minor', 'uint8')]

        nsx_file_id = np.fromfile(filename, count=1, dtype=dt0)[0]
        if nsx_file_id['file_id'].decode() == 'NEURALSG':
            spec = '2.1'
        elif nsx_file_id['file_id'].decode() == 'NEURALCD':
            spec = '{}.{}'.format(
                nsx_file_id['ver_major'], nsx_file_id['ver_minor'])
        else:
            raise IOError('Unsupported NSX file type.')

        return spec

    def __extract_nev_file_spec(self):
        """
        Extract file specification from an .nev file
        """
        filename = '.'.join([self._filenames['nev'], 'nev'])
        # Header structure of files specification 2.2 and higher. For files 2.1
        # and lower, the entries ver_major and ver_minor are not supported.
        dt0 = [
            ('file_id', 'S8'),
            ('ver_major', 'uint8'),
            ('ver_minor', 'uint8')]

        nev_file_id = np.fromfile(filename, count=1, dtype=dt0)[0]
        if nev_file_id['file_id'].decode() == 'NEURALEV':
            spec = '{}.{}'.format(
                nev_file_id['ver_major'], nev_file_id['ver_minor'])
        else:
            raise IOError('NEV file type {} is not supported'.format(
                nev_file_id['file_id']))

        return spec

    def __read_nsx_header_variant_a(self, nsx_nb):
        """
        Extract nsx header information from a 2.1 .nsx file
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        # basic header (file_id: NEURALCD)
        dt0 = [
            ('file_id', 'S8'),
            # label of sampling groun (e.g. "1kS/s" or "LFP Low")
            ('label', 'S16'),
            # number of 1/30000 seconds between data points
            # (e.g., if sampling rate "1 kS/s", period equals "30")
            ('period', 'uint32'),
            ('channel_count', 'uint32')]

        nsx_basic_header = np.fromfile(filename, count=1, dtype=dt0)[0]

        # "extended" header (last field of file_id: NEURALCD)
        # (to facilitate compatibility with higher file specs)
        offset_dt0 = np.dtype(dt0).itemsize
        shape = nsx_basic_header['channel_count']
        # originally called channel_id in Blackrock user manual
        # (to facilitate compatibility with higher file specs)
        dt1 = [('electrode_id', 'uint32')]

        nsx_ext_header = np.memmap(
            filename, shape=shape, offset=offset_dt0, dtype=dt1, mode='r')

        return nsx_basic_header, nsx_ext_header

    def __read_nsx_header_variant_b(self, nsx_nb):
        """
        Extract nsx header information from a 2.2 or 2.3 .nsx file
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        # basic header (file_id: NEURALCD)
        dt0 = [
            ('file_id', 'S8'),
            # file specification split into major and minor version number
            ('ver_major', 'uint8'),
            ('ver_minor', 'uint8'),
            # bytes of basic & extended header
            ('bytes_in_headers', 'uint32'),
            # label of the sampling group (e.g., "1 kS/s" or "LFP low")
            ('label', 'S16'),
            ('comment', 'S256'),
            ('period', 'uint32'),
            ('timestamp_resolution', 'uint32'),
            # time origin: 2byte uint16 values for ...
            ('year', 'uint16'),
            ('month', 'uint16'),
            ('weekday', 'uint16'),
            ('day', 'uint16'),
            ('hour', 'uint16'),
            ('minute', 'uint16'),
            ('second', 'uint16'),
            ('millisecond', 'uint16'),
            # number of channel_count match number of extended headers
            ('channel_count', 'uint32')]

        nsx_basic_header = np.fromfile(filename, count=1, dtype=dt0)[0]

        # extended header (type: CC)
        offset_dt0 = np.dtype(dt0).itemsize
        shape = nsx_basic_header['channel_count']
        dt1 = [
            ('type', 'S2'),
            ('electrode_id', 'uint16'),
            ('electrode_label', 'S16'),
            # used front-end amplifier bank (e.g., A, B, C, D)
            ('physical_connector', 'uint8'),
            # used connector pin (e.g., 1-37 on bank A, B, C or D)
            ('connector_pin', 'uint8'),
            # digital and analog value ranges of the signal
            ('min_digital_val', 'int16'),
            ('max_digital_val', 'int16'),
            ('min_analog_val', 'int16'),
            ('max_analog_val', 'int16'),
            # units of the analog range values ("mV" or "uV")
            ('units', 'S16'),
            # filter settings used to create nsx from source signal
            ('hi_freq_corner', 'uint32'),
            ('hi_freq_order', 'uint32'),
            ('hi_freq_type', 'uint16'),  # 0=None, 1=Butterworth
            ('lo_freq_corner', 'uint32'),
            ('lo_freq_order', 'uint32'),
            ('lo_freq_type', 'uint16')]  # 0=None, 1=Butterworth

        nsx_ext_header = np.memmap(
            filename, shape=shape, offset=offset_dt0, dtype=dt1, mode='r')

        return nsx_basic_header, nsx_ext_header

    def __read_nsx_dataheader(self, nsx_nb, offset):
        """
        Reads data header following the given offset of an nsx file.
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        # dtypes data header
        dt2 = [
            ('header', 'uint8'),
            ('timestamp', 'uint32'),
            ('nb_data_points', 'uint32')]

        return np.memmap(filename, dtype=dt2, shape=1, offset=offset, mode='r')[0]

    def __read_nsx_dataheader_variant_a(
            self, nsx_nb, filesize=None, offset=None):
        """
        Reads None for the nsx data header of file spec 2.1. Introduced to
        facilitate compatibility with higher file spec.
        """

        return None

    def __read_nsx_dataheader_variant_b(
            self, nsx_nb, filesize=None, offset=None, ):
        """
        Reads the nsx data header for each data block following the offset of
        file spec 2.2 and 2.3.
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        filesize = self.__get_file_size(filename)

        data_header = {}
        index = 0

        if offset is None:
            offset = self.__nsx_basic_header[nsx_nb]['bytes_in_headers']

        while offset < filesize:
            dh = self.__read_nsx_dataheader(nsx_nb, offset)
            data_header[index] = {
                'header': dh['header'],
                'timestamp': dh['timestamp'],
                'nb_data_points': dh['nb_data_points'],
                'offset_to_data_block': offset + dh.dtype.itemsize}

            # data size = number of data points * (2bytes * number of channels)
            # use of `int` avoids overflow problem
            data_size = int(dh['nb_data_points']) *\
                int(self.__nsx_basic_header[nsx_nb]['channel_count']) * 2
            # define new offset (to possible next data block)
            offset = data_header[index]['offset_to_data_block'] + data_size

            index += 1

        return data_header

    def __read_nsx_data_variant_a(self, nsx_nb):
        """
        Extract nsx data from a 2.1 .nsx file
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        # get shape of data
        shape = (
            self.__nsx_params['2.1'](nsx_nb)['nb_data_points'],
            self.__nsx_basic_header[nsx_nb]['channel_count'])
        offset = self.__nsx_params['2.1'](nsx_nb)['bytes_in_headers']

        # read nsx data
        # store as dict for compatibility with higher file specs
        data = {0: np.memmap(
            filename, dtype='int16', shape=shape, offset=offset, mode='r')}

        return data

    def __read_nsx_data_variant_b(self, nsx_nb):
        """
        Extract nsx data (blocks) from a 2.2 or 2.3 .nsx file. Blocks can arise
        if the recording was paused by the user.
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        data = {}
        for data_bl in self.__nsx_data_header[nsx_nb].keys():
            # get shape and offset of data
            shape = (
                self.__nsx_data_header[nsx_nb][data_bl]['nb_data_points'],
                self.__nsx_basic_header[nsx_nb]['channel_count'])
            offset = \
                self.__nsx_data_header[nsx_nb][data_bl]['offset_to_data_block']

            # read data
            data[data_bl] = np.memmap(
                filename, dtype='int16', shape=shape, offset=offset, mode='r')

        return data

    def __read_nev_header(self, ext_header_variants):
        """
        Extract nev header information from a of specific .nsx header variant
        """
        filename = '.'.join([self._filenames['nev'], 'nev'])

        # basic header
        dt0 = [
            # Set to "NEURALEV"
            ('file_type_id', 'S8'),
            ('ver_major', 'uint8'),
            ('ver_minor', 'uint8'),
            # Flags
            ('additionnal_flags', 'uint16'),
            # File index of first data sample
            ('bytes_in_headers', 'uint32'),
            # Number of bytes per data packet (sample)
            ('bytes_in_data_packets', 'uint32'),
            # Time resolution of time stamps in Hz
            ('timestamp_resolution', 'uint32'),
            # Sampling frequency of waveforms in Hz
            ('sample_resolution', 'uint32'),
            ('year', 'uint16'),
            ('month', 'uint16'),
            ('weekday', 'uint16'),
            ('day', 'uint16'),
            ('hour', 'uint16'),
            ('minute', 'uint16'),
            ('second', 'uint16'),
            ('millisecond', 'uint16'),
            ('application_to_create_file', 'S32'),
            ('comment_field', 'S256'),
            # Number of extended headers
            ('nb_ext_headers', 'uint32')]

        nev_basic_header = np.fromfile(filename, count=1, dtype=dt0)[0]

        # extended header
        # this consist in N block with code 8bytes + 24 data bytes
        # the data bytes depend on the code and need to be converted
        # cafilename_nsx, segse by case
        shape = nev_basic_header['nb_ext_headers']
        offset_dt0 = np.dtype(dt0).itemsize

        # This is the common structure of the beginning of extended headers
        dt1 = [
            ('packet_id', 'S8'),
            ('info_field', 'S24')]

        raw_ext_header = np.memmap(
            filename, offset=offset_dt0, dtype=dt1, shape=shape, mode='r')

        nev_ext_header = {}
        for packet_id in ext_header_variants.keys():
            mask = (raw_ext_header['packet_id'] == packet_id)
            dt2 = self.__nev_ext_header_types()[packet_id][
                ext_header_variants[packet_id]]

            nev_ext_header[packet_id] = raw_ext_header.view(dt2)[mask]

        return nev_basic_header, nev_ext_header

    def __read_nev_header_variant_a(self):
        """
        Extract nev header information from a 2.1 .nev file
        """

        ext_header_variants = {
            b'NEUEVWAV': 'a',
            b'ARRAYNME': 'a',
            b'ECOMMENT': 'a',
            b'CCOMMENT': 'a',
            b'MAPFILE': 'a',
            b'NSASEXEV': 'a'}

        return self.__read_nev_header(ext_header_variants)

    def __read_nev_header_variant_b(self):
        """
        Extract nev header information from a 2.2 .nev file
        """

        ext_header_variants = {
            b'NEUEVWAV': 'b',
            b'ARRAYNME': 'a',
            b'ECOMMENT': 'a',
            b'CCOMMENT': 'a',
            b'MAPFILE': 'a',
            b'NEUEVLBL': 'a',
            b'NEUEVFLT': 'a',
            b'DIGLABEL': 'a',
            b'NSASEXEV': 'a'}

        return self.__read_nev_header(ext_header_variants)

    def __read_nev_header_variant_c(self):
        """
        Extract nev header information from a 2.3 .nev file
        """

        ext_header_variants = {
            b'NEUEVWAV': 'b',
            b'ARRAYNME': 'a',
            b'ECOMMENT': 'a',
            b'CCOMMENT': 'a',
            b'MAPFILE': 'a',
            b'NEUEVLBL': 'a',
            b'NEUEVFLT': 'a',
            b'DIGLABEL': 'a',
            b'VIDEOSYN': 'a',
            b'TRACKOBJ': 'a'}

        return self.__read_nev_header(ext_header_variants)

    def __read_nev_data(self, nev_data_masks, nev_data_types):
        """
        Extract nev data from a 2.1 or 2.2 .nev file
        """
        filename = '.'.join([self._filenames['nev'], 'nev'])
        data_size = self.__nev_basic_header['bytes_in_data_packets']
        header_size = self.__nev_basic_header['bytes_in_headers']

        # read all raw data packets and markers
        dt0 = [
            ('timestamp', 'uint32'),
            ('packet_id', 'uint16'),
            ('value', 'S{}'.format(data_size - 6))]

        raw_data = np.memmap(filename, offset=header_size, dtype=dt0, mode='r')

        masks = self.__nev_data_masks(raw_data['packet_id'])
        types = self.__nev_data_types(data_size)

        event_segment_ids = self.__get_event_segment_ids(raw_data, masks, nev_data_masks)

        data = {}
        for k, v in nev_data_masks.items():
            mask = masks[k][v]
            data[k] = (raw_data.view(types[k][nev_data_types[k]])[mask], event_segment_ids[mask])

        return data

    def __get_reset_event_mask(self, raw_event_data, masks, nev_data_masks):
        """
        Extract mask for reset comment events in 2.3 .nev file
        """
        restart_mask = np.logical_and(masks['Comments'][nev_data_masks['Comments']],
                                      raw_event_data['value']
                                      == b'\x00\x00\x00\x00\x00\x00critical load restart')
        # TODO: Fix hardcoded number of bytes
        return restart_mask

    def __get_event_segment_ids(self, raw_event_data, masks, nev_data_masks):
        """
        Construct array of corresponding segment ids for each event for nev version 2.3
        """

        if self.__nev_spec in ['2.1', '2.2']:
            # No pause or reset mechanism present for file version 2.1 and 2.2
            return np.zeros(len(raw_event_data), dtype=int)

        elif self.__nev_spec == '2.3':
            reset_ev_mask = self.__get_reset_event_mask(raw_event_data, masks, nev_data_masks)
            reset_ev_ids = np.where(reset_ev_mask)[0]

            # consistency check for monotone increasing time stamps
            # explicitely converting to int to allow for negative diff values
            jump_ids = \
                np.where(np.diff(np.asarray(raw_event_data['timestamp'], dtype=int)) < 0)[0] + 1
            overlap = np.in1d(jump_ids, reset_ev_ids)
            if not all(overlap):
                # additional resets occurred without a reset event being stored
                additional_ids = jump_ids[np.invert(overlap)]
                warnings.warn('Detected {} undocumented segments within '
                              'nev data after timestamps {}.'
                              ''.format(len(additional_ids), additional_ids))
                reset_ev_ids = sorted(np.unique(np.concatenate((reset_ev_ids, jump_ids))))

            event_segment_ids = np.zeros(len(raw_event_data), dtype=int)
            for reset_event_id in reset_ev_ids:
                event_segment_ids[reset_event_id:] += 1

            self._nb_segment_nev = len(reset_ev_ids) + 1
            return event_segment_ids

    def __match_nsx_and_nev_segment_ids(self, nsx_nb):
        """
        Ensure matching ids of segments detected in nsx and nev file for version 2.3
        """

        # NSX required for matching, if not available, warn the user
        if not self._avail_nsx:
            warnings.warn("No nsX available so it cannot be checked whether "
                          "the segments in nev are all correct. Most importantly, "
                          "recording pauses will not be detected", UserWarning)
            return

        # Only needs to be done for nev version 2.3
        if self.__nev_spec == '2.3':
            nsx_offset = self.__nsx_data_header[nsx_nb][0]['timestamp']
            # Multiples of 1/30.000s that pass between two nsX samples
            nsx_period = self.__nsx_basic_header[nsx_nb]['period']
            # NSX segments needed as dict and list
            nonempty_nsx_segments = {}
            list_nonempty_nsx_segments = []
            # Counts how many segments CAN be created from nev
            nb_possible_nev_segments = self._nb_segment_nev

            # Nonempty segments are those containing at least 2 samples
            # These have to be able to be mapped to nev
            for k, v in sorted(self.__nsx_data_header[nsx_nb].items()):
                if v['nb_data_points'] > 1:
                    nonempty_nsx_segments[k] = v
                    list_nonempty_nsx_segments.append(v)

            # Account for paused segments
            # This increases nev event segment ids if from the nsx an additional segment is found
            # If one new segment, i.e. that could not be determined from the nev was found,
            # all following ids need to be increased to account for the additional segment before
            for k, (data, ev_ids) in self.nev_data.items():

                # Check all nonempty nsX segments
                for i, seg in enumerate(list_nonempty_nsx_segments[:]):

                    # Last timestamp in this nsX segment
                    # Not subtracting nsX offset from end because spike extraction might continue
                    end_of_current_nsx_seg = seg['timestamp'] + \
                        seg['nb_data_points'] * self.__nsx_basic_header[nsx_nb]['period']

                    mask_after_seg = (ev_ids == i) & \
                                    (data['timestamp'] > end_of_current_nsx_seg + nsx_period)

                    # Show warning if spikes do not fit any segment (+- 1 sampling 'tick')
                    # Spike should belong to segment before
                    mask_outside = (ev_ids == i) & \
                            (data['timestamp'] < int(seg['timestamp']) - nsx_offset - nsx_period)

                    if len(data[mask_outside]) > 0:
                        warnings.warn("Spikes outside any segment. Detected on segment #{}".
                                      format(i))
                        ev_ids[mask_outside] -= 1

                    # If some nev data are outside of this nsX segment, increase their segment ids
                    # and the ids of all following segments. They are checked for the next nsX
                    # segment then. If they do not fit any of them,
                    # a warning will be shown, indicating how far outside the segment spikes are
                    # If they fit the next segment, more segments are possible in nev,
                    # because a new one has been discovered
                    if len(data[mask_after_seg]) > 0:
                        # Warning if spikes are after last segment
                        if i == len(list_nonempty_nsx_segments) - 1:
                            timestamp_resolution = self.__nsx_params[self.__nsx_spec[
                                nsx_nb]]('timestamp_resolution', nsx_nb)
                            time_after_seg = (data[mask_after_seg]['timestamp'][-1]
                                              - end_of_current_nsx_seg) / timestamp_resolution
                            warnings.warn("Spikes {}s after last segment.".format(time_after_seg))
                            # Break out of loop because it's the last iteration
                            # and the spikes should stay connected to last segment
                            break

                        # If reset and no segment detected in nev, then these segments cannot be
                        # distinguished in nev, which is a big problem
                        # XXX 96 is an arbitrary number based on observations in available files
                        elif list_nonempty_nsx_segments[i + 1]['timestamp'] - nsx_offset <= 96:
                            # If not all definitely belong to the next segment,
                            # then it cannot be distinguished where some belong
                            if len(data[ev_ids == i]) != len(data[mask_after_seg]):
                                raise ValueError("Some segments in nsX cannot be detected in nev")

                        # Actual processing if no problem has occurred
                        nb_possible_nev_segments += 1
                        ev_ids[ev_ids > i] += 1
                        ev_ids[mask_after_seg] += 1

            # consistency check: same number of segments for nsx and nev data
            assert nb_possible_nev_segments == len(nonempty_nsx_segments), \
                ('Inconsistent ns{0} and nev file. {1} segments present in .nev file, but {2} in '
                 'ns{0} file.'.format(nsx_nb, nb_possible_nev_segments,
                                      len(nonempty_nsx_segments)))

            new_nev_segment_id_mapping = dict(zip(range(nb_possible_nev_segments),
                                                  sorted(list(nonempty_nsx_segments))))

            # replacing event ids by matched event ids in place
            for k, (data, ev_ids) in self.nev_data.items():
                if len(ev_ids):
                    ev_ids[:] = np.vectorize(new_nev_segment_id_mapping.__getitem__)(ev_ids)

    def __read_nev_data_variant_a(self):
        """
        Extract nev data from a 2.1 & 2.2 .nev file
        """
        nev_data_masks = {
            'NonNeural': 'a',
            'Spikes': 'a'}

        nev_data_types = {
            'NonNeural': 'a',
            'Spikes': 'a'}

        return self.__read_nev_data(nev_data_masks, nev_data_types)

    def __read_nev_data_variant_b(self):
        """
        Extract nev data from a 2.3 .nev file
        """
        nev_data_masks = {
            'NonNeural': 'a',
            'Spikes': 'b',
            'Comments': 'a',
            'VideoSync': 'a',
            'TrackingEvents': 'a',
            'ButtonTrigger': 'a',
            'ConfigEvent': 'a'}

        nev_data_types = {
            'NonNeural': 'b',
            'Spikes': 'a',
            'Comments': 'a',
            'VideoSync': 'a',
            'TrackingEvents': 'a',
            'ButtonTrigger': 'a',
            'ConfigEvent': 'a'}

        return self.__read_nev_data(nev_data_masks, nev_data_types)

    def __nev_ext_header_types(self):
        """
        Defines extended header types for different .nev file specifications.
        """
        nev_ext_header_types = {
            b'NEUEVWAV': {
                # Version>=2.1
                'a': [
                    ('packet_id', 'S8'),
                    ('electrode_id', 'uint16'),
                    ('physical_connector', 'uint8'),
                    ('connector_pin', 'uint8'),
                    ('digitization_factor', 'uint16'),
                    ('energy_threshold', 'uint16'),
                    ('hi_threshold', 'int16'),
                    ('lo_threshold', 'int16'),
                    ('nb_sorted_units', 'uint8'),
                    # number of bytes per waveform sample
                    ('bytes_per_waveform', 'uint8'),
                    ('unused', 'S10')],
                # Version>=2.3
                'b': [
                    ('packet_id', 'S8'),
                    ('electrode_id', 'uint16'),
                    ('physical_connector', 'uint8'),
                    ('connector_pin', 'uint8'),
                    ('digitization_factor', 'uint16'),
                    ('energy_threshold', 'uint16'),
                    ('hi_threshold', 'int16'),
                    ('lo_threshold', 'int16'),
                    ('nb_sorted_units', 'uint8'),
                    # number of bytes per waveform sample
                    ('bytes_per_waveform', 'uint8'),
                    # number of samples for each waveform
                    ('spike_width', 'uint16'),
                    ('unused', 'S8')]},
            b'ARRAYNME': {
                'a': [
                    ('packet_id', 'S8'),
                    ('electrode_array_name', 'S24')]},
            b'ECOMMENT': {
                'a': [
                    ('packet_id', 'S8'),
                    ('extra_comment', 'S24')]},
            b'CCOMMENT': {
                'a': [
                    ('packet_id', 'S8'),
                    ('continued_comment', 'S24')]},
            b'MAPFILE': {
                'a': [
                    ('packet_id', 'S8'),
                    ('mapFile', 'S24')]},
            b'NEUEVLBL': {
                'a': [
                    ('packet_id', 'S8'),
                    ('electrode_id', 'uint16'),
                    # label of this electrode
                    ('label', 'S16'),
                    ('unused', 'S6')]},
            b'NEUEVFLT': {
                'a': [
                    ('packet_id', 'S8'),
                    ('electrode_id', 'uint16'),
                    ('hi_freq_corner', 'uint32'),
                    ('hi_freq_order', 'uint32'),
                    # 0=None 1=Butterworth
                    ('hi_freq_type', 'uint16'),
                    ('lo_freq_corner', 'uint32'),
                    ('lo_freq_order', 'uint32'),
                    # 0=None 1=Butterworth
                    ('lo_freq_type', 'uint16'),
                    ('unused', 'S2')]},
            b'DIGLABEL': {
                'a': [
                    ('packet_id', 'S8'),
                    # Read name of digital
                    ('label', 'S16'),
                    # 0=serial, 1=parallel
                    ('mode', 'uint8'),
                    ('unused', 'S7')]},
            b'NSASEXEV': {
                'a': [
                    ('packet_id', 'S8'),
                    # Read frequency of periodic packet generation
                    ('frequency', 'uint16'),
                    # Read if digital input triggers events
                    ('digital_input_config', 'uint8'),
                    # Read if analog input triggers events
                    ('analog_channel_1_config', 'uint8'),
                    ('analog_channel_1_edge_detec_val', 'uint16'),
                    ('analog_channel_2_config', 'uint8'),
                    ('analog_channel_2_edge_detec_val', 'uint16'),
                    ('analog_channel_3_config', 'uint8'),
                    ('analog_channel_3_edge_detec_val', 'uint16'),
                    ('analog_channel_4_config', 'uint8'),
                    ('analog_channel_4_edge_detec_val', 'uint16'),
                    ('analog_channel_5_config', 'uint8'),
                    ('analog_channel_5_edge_detec_val', 'uint16'),
                    ('unused', 'S6')]},
            b'VIDEOSYN': {
                'a': [
                    ('packet_id', 'S8'),
                    ('video_source_id', 'uint16'),
                    ('video_source', 'S16'),
                    ('frame_rate', 'float32'),
                    ('unused', 'S2')]},
            b'TRACKOBJ': {
                'a': [
                    ('packet_id', 'S8'),
                    ('trackable_type', 'uint16'),
                    ('trackable_id', 'uint16'),
                    ('point_count', 'uint16'),
                    ('video_source', 'S16'),
                    ('unused', 'S2')]}}

        return nev_ext_header_types

    def __nev_data_masks(self, packet_ids):
        """
        Defines data masks for different .nev file specifications depending on
        the given packet identifiers.
        """
        __nev_data_masks = {
            'NonNeural': {
                'a': (packet_ids == 0)},
            'Spikes': {
                # Version 2.1 & 2.2
                'a': (0 < packet_ids) & (packet_ids <= 255),
                # Version>=2.3
                'b': (0 < packet_ids) & (packet_ids <= 2048)},
            'Comments': {
                'a': (packet_ids == 0xFFFF)},
            'VideoSync': {
                'a': (packet_ids == 0xFFFE)},
            'TrackingEvents': {
                'a': (packet_ids == 0xFFFD)},
            'ButtonTrigger': {
                'a': (packet_ids == 0xFFFC)},
            'ConfigEvent': {
                'a': (packet_ids == 0xFFFB)}}

        return __nev_data_masks

    def __nev_data_types(self, data_size):
        """
        Defines data types for different .nev file specifications depending on
        the given packet identifiers.
        """
        __nev_data_types = {
            'NonNeural': {
                # Version 2.1 & 2.2
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('packet_insertion_reason', 'uint8'),
                    ('reserved', 'uint8'),
                    ('digital_input', 'uint16'),
                    ('analog_input_channel_1', 'int16'),
                    ('analog_input_channel_2', 'int16'),
                    ('analog_input_channel_3', 'int16'),
                    ('analog_input_channel_4', 'int16'),
                    ('analog_input_channel_5', 'int16'),
                    ('unused', 'S{}'.format(data_size - 20))],
                # Version>=2.3
                'b': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('packet_insertion_reason', 'uint8'),
                    ('reserved', 'uint8'),
                    ('digital_input', 'uint16'),
                    ('unused', 'S{}'.format(data_size - 10))]},
            'Spikes': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('unit_class_nb', 'uint8'),
                    ('reserved', 'uint8'),
                    ('waveform', 'S{}'.format(data_size - 8))]},
            'Comments': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('char_set', 'uint8'),
                    ('flag', 'uint8'),
                    ('color', 'uint32'),
                    ('comment', 'S{}'.format(data_size - 12))]},
            'VideoSync': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('video_file_nb', 'uint16'),
                    ('video_frame_nb', 'uint32'),
                    ('video_elapsed_time', 'uint32'),
                    ('video_source_id', 'uint32'),
                    ('unused', 'int8', (data_size - 20,))]},
            'TrackingEvents': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('parent_id', 'uint16'),
                    ('node_id', 'uint16'),
                    ('node_count', 'uint16'),
                    ('point_count', 'uint16'),
                    ('tracking_points', 'uint16', ((data_size - 14) // 2,))]},
            'ButtonTrigger': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('trigger_type', 'uint16'),
                    ('unused', 'int8', (data_size - 8,))]},
            'ConfigEvent': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('config_change_type', 'uint16'),
                    ('config_changed', 'S{}'.format(data_size - 8))]}}

        return __nev_data_types

    def __nev_params(self, param_name):
        """
        Returns wanted nev parameter.
        """
        nev_parameters = {
            'bytes_in_data_packets':
                self.__nev_basic_header['bytes_in_data_packets'],
            'rec_datetime': datetime.datetime(
                year=self.__nev_basic_header['year'],
                month=self.__nev_basic_header['month'],
                day=self.__nev_basic_header['day'],
                hour=self.__nev_basic_header['hour'],
                minute=self.__nev_basic_header['minute'],
                second=self.__nev_basic_header['second'],
                microsecond=self.__nev_basic_header['millisecond']),
            'max_res': self.__nev_basic_header['timestamp_resolution'],
            'channel_ids': self.__nev_ext_header[b'NEUEVWAV']['electrode_id'],
            'channel_labels': self.__channel_labels[self.__nev_spec](),
            'event_unit': pq.CompoundUnit("1.0/{} * s".format(
                self.__nev_basic_header['timestamp_resolution'])),
            'nb_units': dict(zip(
                self.__nev_ext_header[b'NEUEVWAV']['electrode_id'],
                self.__nev_ext_header[b'NEUEVWAV']['nb_sorted_units'])),
            'digitization_factor': dict(zip(
                self.__nev_ext_header[b'NEUEVWAV']['electrode_id'],
                self.__nev_ext_header[b'NEUEVWAV']['digitization_factor'])),
            'data_size': self.__nev_basic_header['bytes_in_data_packets'],
            'waveform_size': self.__waveform_size[self.__nev_spec](),
            'waveform_dtypes': self.__get_waveforms_dtype(),
            'waveform_sampling_rate':
                self.__nev_basic_header['sample_resolution'] * pq.Hz,
            'waveform_time_unit': pq.CompoundUnit("1.0/{} * s".format(
                self.__nev_basic_header['sample_resolution'])),
            'waveform_unit': pq.uV}

        return nev_parameters[param_name]

    def __get_file_size(self, filename):
        """
        Returns the file size in bytes for the given file.
        """
        filebuf = open(filename, 'rb')
        filebuf.seek(0, os.SEEK_END)
        file_size = filebuf.tell()
        filebuf.close()

        return file_size

    def __get_min_time(self):
        """
        Returns the smallest time that can be determined from the recording for
        use as the lower bound n in an interval [n,m).
        """
        tp = []
        if self._avail_files['nev']:
            tp.extend(self.__get_nev_rec_times()[0])
        for nsx_i in self._avail_nsx:
            tp.extend(self.__nsx_rec_times[self.__nsx_spec[nsx_i]](nsx_i)[0])

        return min(tp)

    def __get_max_time(self):
        """
        Returns the largest time that can be determined from the recording for
        use as the upper bound m in an interval [n,m).
        """
        tp = []
        if self._avail_files['nev']:
            tp.extend(self.__get_nev_rec_times()[1])
        for nsx_i in self._avail_nsx:
            tp.extend(self.__nsx_rec_times[self.__nsx_spec[nsx_i]](nsx_i)[1])

        return max(tp)

    def __get_nev_rec_times(self):
        """
        Extracts minimum and maximum time points from a nev file.
        """
        filename = '.'.join([self._filenames['nev'], 'nev'])

        dt = [('timestamp', 'uint32')]
        offset = \
            self.__get_file_size(filename) - \
            self.__nev_params('bytes_in_data_packets')
        last_data_packet = np.memmap(filename, offset=offset, dtype=dt, mode='r')[0]

        n_starts = [0 * self.__nev_params('event_unit')]
        n_stops = [
            last_data_packet['timestamp'] * self.__nev_params('event_unit')]

        return n_starts, n_stops

    def __get_waveforms_dtype(self):
        """
        Extracts the actual waveform dtype set for each channel.
        """
        # Blackrock code giving the approiate dtype
        conv = {0: 'int8', 1: 'int8', 2: 'int16', 4: 'int32'}

        # get all electrode ids from nev ext header
        all_el_ids = self.__nev_ext_header[b'NEUEVWAV']['electrode_id']

        # get the dtype of waveform (this is stupidly complicated)
        if self.__is_set(
                np.array(self.__nev_basic_header['additionnal_flags']), 0):
            dtype_waveforms = {k: 'int16' for k in all_el_ids}
        else:
            # extract bytes per waveform
            waveform_bytes = \
                self.__nev_ext_header[b'NEUEVWAV']['bytes_per_waveform']
            # extract dtype for waveforms fro each electrode
            dtype_waveforms = dict(zip(all_el_ids, conv[waveform_bytes]))

        return dtype_waveforms

    def __get_channel_labels_variant_a(self):
        """
        Returns labels for all channels for file spec 2.1
        """
        elids = self.__nev_ext_header[b'NEUEVWAV']['electrode_id']
        labels = []

        for elid in elids:
            if elid < 129:
                labels.append('chan%i' % elid)
            else:
                labels.append('ainp%i' % (elid - 129 + 1))

        return dict(zip(elids, labels))

    def __get_channel_labels_variant_b(self):
        """
        Returns labels for all channels for file spec 2.2 and 2.3
        """
        elids = self.__nev_ext_header[b'NEUEVWAV']['electrode_id']
        labels = self.__nev_ext_header[b'NEUEVLBL']['label']

        return dict(zip(elids, labels)) if len(labels) > 0 else None

    def __get_waveform_size_variant_a(self):
        """
        Returns wavform sizes for all channels for file spec 2.1 and 2.2
        """
        wf_dtypes = self.__get_waveforms_dtype()
        nb_bytes_wf = self.__nev_basic_header['bytes_in_data_packets'] - 8

        wf_sizes = {
            ch: int(nb_bytes_wf / np.dtype(dt).itemsize) for ch, dt in
            wf_dtypes.items()}

        return wf_sizes

    def __get_waveform_size_variant_b(self):
        """
        Returns wavform sizes for all channels for file spec 2.3
        """
        elids = self.__nev_ext_header[b'NEUEVWAV']['electrode_id']
        spike_widths = self.__nev_ext_header[b'NEUEVWAV']['spike_width']

        return dict(zip(elids, spike_widths))

    def __get_left_sweep_waveforms(self):
        """
        Returns left sweep of waveforms for each channel. Left sweep is defined
        as the time from the beginning of the waveform to the trigger time of
        the corresponding spike.
        """
        # TODO: Double check if this is the actual setting for Blackrock
        wf_t_unit = self.__nev_params('waveform_time_unit')
        all_ch = self.__nev_params('channel_ids')

        # TODO: Double check if this is the correct assumption (10 samples)
        # default value: threshold crossing after 10 samples of waveform
        wf_left_sweep = {ch: 10 * wf_t_unit for ch in all_ch}

        # non-default: threshold crossing at center of waveform
        # wf_size = self.__nev_params('waveform_size')
        # wf_left_sweep = dict(
        #     [(ch, (wf_size[ch] / 2) * wf_t_unit) for ch in all_ch])

        return wf_left_sweep

    def __get_nsx_param_variant_a(self, nsx_nb):
        """
        Returns parameter (param_name) for a given nsx (nsx_nb) for file spec
        2.1.
        """
        # Here, min/max_analog_val and min/max_digital_val are not available in
        # the nsx, so that we must estimate these parameters from the
        # digitization factor of the nev (information by Kian Torab, Blackrock
        # Microsystems). Here dig_factor=max_analog_val/max_digital_val. We set
        # max_digital_val to 1000, and max_analog_val=dig_factor. dig_factor is
        # given in nV by definition, so the units turn out to be uV.
        labels = []
        dig_factor = []
        for elid in self.__nsx_ext_header[nsx_nb]['electrode_id']:
            if self._avail_files['nev']:
                # This is a workaround for the DigitalFactor overflow in NEV
                # files recorded with buggy Cerebus system.
                # Fix taken from: NMPK toolbox by Blackrock,
                # file openNEV, line 464,
                # git rev. d0a25eac902704a3a29fa5dfd3aed0744f4733ed
                df = self.__nev_params('digitization_factor')[elid]
                if df == 21516:
                    df = 152592.547
                dig_factor.append(df)
            else:
                dig_factor.append(float('nan'))

            if elid < 129:
                labels.append('chan%i' % elid)
            else:
                labels.append('ainp%i' % (elid - 129 + 1))

        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        bytes_in_headers = self.__nsx_basic_header[nsx_nb].dtype.itemsize + \
            self.__nsx_ext_header[nsx_nb].dtype.itemsize * \
            self.__nsx_basic_header[nsx_nb]['channel_count']

        if np.isnan(dig_factor[0]):
            units = ''
            warnings.warn("Cannot rescale to voltage, raw data will be returned.", UserWarning)
        else:
            units = 'uV'

        nsx_parameters = {
            'nb_data_points': int(
                (self.__get_file_size(filename) - bytes_in_headers)
                / (2 * self.__nsx_basic_header[nsx_nb]['channel_count']) - 1),
            'labels': labels,
            'units': np.array([units] * self.__nsx_basic_header[nsx_nb]['channel_count']),
            'min_analog_val': -1 * np.array(dig_factor),
            'max_analog_val': np.array(dig_factor),
            'min_digital_val': np.array(
                [-1000] * self.__nsx_basic_header[nsx_nb]['channel_count']),
            'max_digital_val': np.array([1000] * self.__nsx_basic_header[nsx_nb]['channel_count']),
            'timestamp_resolution': 30000,
            'bytes_in_headers': bytes_in_headers,
            'sampling_rate': 30000 / self.__nsx_basic_header[nsx_nb]['period'] * pq.Hz,
            'time_unit': pq.CompoundUnit("1.0/{}*s".format(
                30000 / self.__nsx_basic_header[nsx_nb]['period']))}

        # Returns complete dictionary because then it does not need to be called so often
        return nsx_parameters

    def __get_nsx_param_variant_b(self, param_name, nsx_nb):
        """
        Returns parameter (param_name) for a given nsx (nsx_nb) for file spec
        2.2 and 2.3.
        """
        nsx_parameters = {
            'labels':
                self.__nsx_ext_header[nsx_nb]['electrode_label'],
            'units':
                self.__nsx_ext_header[nsx_nb]['units'],
            'min_analog_val':
                self.__nsx_ext_header[nsx_nb]['min_analog_val'],
            'max_analog_val':
                self.__nsx_ext_header[nsx_nb]['max_analog_val'],
            'min_digital_val':
                self.__nsx_ext_header[nsx_nb]['min_digital_val'],
            'max_digital_val':
                self.__nsx_ext_header[nsx_nb]['max_digital_val'],
            'timestamp_resolution':
                self.__nsx_basic_header[nsx_nb]['timestamp_resolution'],
            'bytes_in_headers':
                self.__nsx_basic_header[nsx_nb]['bytes_in_headers'],
            'sampling_rate':
                self.__nsx_basic_header[nsx_nb]['timestamp_resolution']
                / self.__nsx_basic_header[nsx_nb]['period'] * pq.Hz,
            'time_unit': pq.CompoundUnit("1.0/{}*s".format(
                self.__nsx_basic_header[nsx_nb]['timestamp_resolution']
                / self.__nsx_basic_header[nsx_nb]['period']))}

        return nsx_parameters[param_name]

    def __get_nonneural_evdicts_variant_a(self, data):
        """
        Defines event types and the necessary parameters to extract them from
        a 2.1 and 2.2 nev file.
        """
        # TODO: add annotations of nev ext header (NSASEXEX) to event types

        # digital events
        event_types = {
            'digital_input_port': {
                'name': 'digital_input_port',
                'field': 'digital_input',
                'mask': data['packet_insertion_reason'] == 1,
                'desc': "Events of the digital input port"},
            'serial_input_port': {
                'name': 'serial_input_port',
                'field': 'digital_input',
                'mask': data['packet_insertion_reason'] == 129,
                'desc': "Events of the serial input port"}}

        # analog input events via threshold crossings
        for ch in range(5):
            event_types.update({
                'analog_input_channel_{}'.format(ch + 1): {
                    'name': 'analog_input_channel_{}'.format(ch + 1),
                    'field': 'analog_input_channel_{}'.format(ch + 1),
                    'mask': self.__is_set(
                        data['packet_insertion_reason'], ch + 1),
                    'desc': "Values of analog input channel {} in mV "
                            "(+/- 5000)".format(ch + 1)}})

        # TODO: define field and desc
        event_types.update({
            'periodic_sampling_events': {
                'name': 'periodic_sampling_events',
                'field': 'digital_input',
                'mask': self.__is_set(data['packet_insertion_reason'], 6),
                'desc': 'Periodic sampling event of a certain frequency'}})

        return event_types

    def __delete_empty_segments(self):
        """
        If there are empty segments (e.g. due to a reset or clock synchronization across
        two systems), these can be discarded.
        If this is done, all the data and data_headers need to be remapped to become a range
        starting from 0 again. Nev data are mapped accordingly to stay with their corresponding
        segment in the nsX data.
        """

        # Discard empty segments
        removed_seg = []
        for data_bl in range(self._nb_segment):
            keep_seg = True
            for nsx_nb in self.nsx_to_load:
                length = self.nsx_datas[nsx_nb][data_bl].shape[0]
                keep_seg = keep_seg and (length >= 2)

            if not keep_seg:
                removed_seg.append(data_bl)
                for nsx_nb in self.nsx_to_load:
                    self.nsx_datas[nsx_nb].pop(data_bl)
                    self.__nsx_data_header[nsx_nb].pop(data_bl)

        # Keys need to be increasing from 0 to maximum in steps of 1
        # To ensure this after removing empty segments, some keys need to be re mapped
        for i in removed_seg[::-1]:
            for j in range(i + 1, self._nb_segment):
                # remap nsx seg index
                for nsx_nb in self.nsx_to_load:
                    data = self.nsx_datas[nsx_nb].pop(j)
                    self.nsx_datas[nsx_nb][j - 1] = data

                    data_header = self.__nsx_data_header[nsx_nb].pop(j)
                    self.__nsx_data_header[nsx_nb][j - 1] = data_header

                # Also remap nev data, ev_ids are the equivalent to keys above
                if self._avail_files['nev']:
                    for k, (data, ev_ids) in self.nev_data.items():
                        ev_ids[ev_ids == j] -= 1

            self._nb_segment -= 1

    def __get_nonneural_evdicts_variant_b(self, data):
        """
        Defines event types and the necessary parameters to extract them from
        a 2.3 nev file.
        """
        # digital events
        if not np.all(np.in1d(data['packet_insertion_reason'], [1, 129])):
            # Blackrock spec gives reason==64 means PERIODIC, but never seen this live
            warnings.warn("Unknown event codes found", RuntimeWarning)
        event_types = {
            'digital_input_port': {
                'name': 'digital_input_port',
                'field': 'digital_input',
                'mask': self.__is_set(data['packet_insertion_reason'], 0)
                        & ~self.__is_set(data['packet_insertion_reason'], 7),
                'desc': "Events of the digital input port"},
            'serial_input_port': {
                'name': 'serial_input_port',
                'field': 'digital_input',
                'mask': self.__is_set(data['packet_insertion_reason'], 0)
                        & self.__is_set(data['packet_insertion_reason'], 7),
                'desc': "Events of the serial input port"}}

        return event_types

    def __get_comment_evdict_variant_a(self, data):
        return {
            'comments': {
                'name': 'comments',
                'field': 'comment',
                'mask': data['packet_id'] == 65535,
                'desc': 'Comments'
            }
        }

    def __is_set(self, flag, pos):
        """
        Checks if bit is set at the given position for flag. If flag is an
        array, an array will be returned.
        """
        return flag & (1 << pos) > 0
