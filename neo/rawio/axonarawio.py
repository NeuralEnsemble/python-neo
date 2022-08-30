"""
This class reads .set and .bin file data from the Axona acquisition system.

File format overview:
http://space-memory-navigation.org/DacqUSBFileFormats.pdf

In brief:
 data.set - setup file containing all hardware setups related to the trial
 data.bin - raw data file

There are many other data formats from Axona, which we do not consider (yet).
These are derivative from the raw continuous data (.bin) and could in principle
be extracted from it (see file format overview for details).

Author: Steffen Buergers, Julia Sprenger

"""

import datetime
import pathlib
import re
import numpy as np

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)


class AxonaRawIO(BaseRawIO):
    """
    Class for reading raw, continuous data from the Axona dacqUSB system:
    http://space-memory-navigation.org/DacqUSBFileFormats.pdf

    The raw data is saved in .bin binary files with an accompanying .set
    file about the recording setup (see the above manual for details).

    Usage::

        import neo.rawio
        r = neo.rawio.AxonaRawIO(filename=os.path.join(dir_name, base_filename))
        r.parse_header()
        print(r)
        raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
                                             i_start=0, i_stop=1024,
                                             channel_names=channel_names)
        float_chunk = reader.rescale_signal_raw_to_float(
            raw_chunk, dtype='float64',
            channel_indexes=[0, 3, 6]
        )

    """

    extensions = ['bin', 'set'] + [str(i) for i in range(1, 33)]  # Never used?
    rawmode = 'multi-file'

    # In the .bin file, channels are arranged in a strange order.
    # This list takes a channel index as input and returns the actual
    # offset for the channel in the memory map (self._raw_signals).
    channel_memory_offset = [
        32, 33, 34, 35, 36, 37, 38, 39, 0, 1, 2, 3, 4, 5,
        6, 7, 40, 41, 42, 43, 44, 45, 46, 47, 8, 9, 10, 11,
        12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 16, 17,
        18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63,
        24, 25, 26, 27, 28, 29, 30, 31
    ]

    def __init__(self, filename):
        BaseRawIO.__init__(self)

        # Accepting filename with arbitrary suffix as input
        self.filename = pathlib.Path(filename).with_suffix('')
        self.set_file = self.filename.with_suffix('.set')
        self.bin_file = None
        self.tetrode_files = []

        # set file is mandatory for any recording
        if not self.set_file.exists():
            raise ValueError(f'Could not locate ".set" file. '
                             f'{self.filename.with_suffix(".set")} does not '
                             f'exist.')

        # detecting available files
        if self.filename.with_suffix('.bin').exists():
            self.bin_file = self.filename.with_suffix('.bin')

        for i in range(1, 33):
            unit_file = self.filename.with_suffix(f'.{i}')
            if unit_file.exists():
                self.tetrode_files.append(unit_file)
            else:
                break

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        '''
        Read important information from .set header file, create memory map
        to raw data (.bin file) and prepare header dictionary in neo format.
        '''

        unit_dtype = np.dtype([('spiketimes', '>i4'),
                               ('samples', 'int8', (50,))])

        # Utility collection of file parameters (general info and header data)
        params = {'bin': {'filename': self.bin_file,
                          'bytes_packet': 432,
                          'bytes_data': 384,
                          'bytes_head': 32,
                          'bytes_tail': 16,
                          'data_type': 'int16',
                          'header_size': 0,
                          # bin files don't contain a file header
                          'header_encoding': None},
                  'set': {'filename': self.set_file,
                          'header_encoding': 'cp1252'},
                  'unit': {'data_type': unit_dtype,
                           'tetrode_ids': [],
                           'header_encoding': 'cp1252'}}
        self.file_parameters = params

        # SCAN SET FILE
        set_dict = self.get_header_parameters(self.set_file, 'set')
        params['set']['file_header'] = set_dict
        params['set']['sampling_rate'] = int(set_dict['rawRate'])

        # SCAN BIN FILE
        signal_streams = []
        signal_channels = []
        if self.bin_file:
            bin_dict = self.file_parameters['bin']
            # add derived parameters from bin file
            bin_dict['num_channels'] = len(self.get_active_tetrode()) * 4
            num_tot_packets = int(self.bin_file.stat().st_size
                                  / bin_dict['bytes_packet'])
            bin_dict['num_total_packets'] = num_tot_packets
            bin_dict['num_total_samples'] = num_tot_packets * 3

            # Create np.memmap to .bin file
            self._raw_signals = np.memmap(
                self.bin_file, dtype=self.file_parameters['bin']['data_type'],
                mode='r', offset=self.file_parameters['bin']['header_size']
            )

            signal_streams = self._get_signal_streams_header()
            signal_channels = self._get_signal_chan_header()

        # SCAN TETRODE FILES
        # In this IO one tetrode corresponds to one unit as spikes are not
        # sorted yet.
        self._raw_spikes = []
        spike_channels = []
        if self.tetrode_files:

            for i, tetrode_file in enumerate(self.tetrode_files):
                # collecting tetrode specific parameters and dtype conversions
                tdict = self.get_header_parameters(tetrode_file, 'unit')
                tdict['filename'] = tetrode_file
                tdict['num_chans'] = int(tdict['num_chans'])
                tdict['num_spikes'] = int(tdict['num_spikes'])
                tdict['header_size'] = len(
                    self.get_header_bstring(tetrode_file))

                # memory mapping spiking data
                spikes = np.memmap(
                    tetrode_file,
                    dtype=self.file_parameters['unit']['data_type'],
                    mode='r', offset=tdict['header_size'],
                    shape=(tdict['num_spikes'], 4))
                self._raw_spikes.append(spikes)

                unit_name = f'tetrode {i + 1}'
                unit_id = f'{i + 1}'
                wf_units = 'dimensionless'
                wf_gain = 1
                wf_offset = 0.
                # left sweep information is only stored in set file
                wf_left_sweep = int(self.file_parameters['set']['file_header']
                                    ['pretrigSamps'])

                # Extract waveform sample rate
                # 1st priority source: Spike2msMode (0 -> 48kHz; 1 -> 24kHz)
                # 2nd priority source: tetrode sample rate
                spikemode_to_sr = {0: 48000, 1: 24000}  # spikemode->rate in Hz
                sm = self.file_parameters['set']['file_header'].get(
                    'Spike2msMode', -1)
                wf_sampling_rate = spikemode_to_sr.get(int(sm), None)
                if wf_sampling_rate is None:
                    wf_sampling_rate = self._to_hz(tdict['sample_rate'],
                                               dtype=float)

                spike_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                       wf_offset, wf_left_sweep,
                                       wf_sampling_rate))

                self.file_parameters['unit']['tetrode_ids'].append(i + 1)
                self.file_parameters['unit'][i + 1] = tdict

            # propagate common tetrode parameters to global unit level
            units_dict = self.file_parameters['unit']
            ids = units_dict['tetrode_ids']
            copied_keys = []
            if ids:
                for key, value in units_dict[ids[0]].items():
                    # copy key-value pair if present across all tetrodes
                    if all([key in units_dict[t] for t in ids]) and \
                            all([units_dict[t][key] == value for t in ids]):
                        self.file_parameters['unit'][key] = value
                        copied_keys.append(key)

                # remove key from individual tetrode parameters
                for key in copied_keys:
                    for t in ids:
                        self.file_parameters['unit'][t].pop(key)

        # Create RawIO header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = np.array(signal_streams,
                                                 dtype=_signal_stream_dtype)
        self.header['signal_channels'] = np.array(signal_channels,
                                                  dtype=_signal_channel_dtype)
        self.header['spike_channels'] = np.array(spike_channels,
                                                 dtype=_spike_channel_dtype)
        self.header['event_channels'] = np.array([],
                                                 dtype=_event_channel_dtype)

        # Annotations
        self._generate_minimal_annotations()

        # Adding custom annotations
        bl_ann = self.raw_annotations['blocks'][0]
        seg_ann = bl_ann['segments'][0]
        seg_ann['rec_datetime'] = self.read_datetime()
        if len(seg_ann['signals']):
            seg_ann['signals'][0]['__array_annotations__']['tetrode_id'] = \
                [tetr for tetr in self.get_active_tetrode() for _ in range(4)]

        if len(seg_ann['spikes']):
            # adding segment annotations
            seg_keys = ['experimenter', 'comments', 'sw_version']
            for seg_key in seg_keys:
                if seg_key in self.file_parameters['unit']:
                    seg_ann[seg_key] = self.file_parameters['unit'][seg_key]

    def _get_signal_streams_header(self):
        # create signals stream information (we always expect a single stream)
        return np.array([('stream 0', '0')], dtype=_signal_stream_dtype)

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = 0.

        if 'num_total_packets' in self.file_parameters['bin']:
            sr = self.file_parameters['set']['sampling_rate']
            t_stop = self.file_parameters['bin']['num_total_samples'] / sr

        if self.file_parameters['unit']['tetrode_ids']:
            # get tetrode recording durations in seconds
            if 'duration' not in self.file_parameters['unit']:
                raise ValueError('Can not determine common tetrode recording'
                                 'duration.')

            tetrode_duration = float(self.file_parameters['unit']['duration'])
            t_stop = max(t_stop, tetrode_duration)

        return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        if 'num_total_packets' in self.file_parameters['bin']:
            return self.file_parameters['bin']['num_total_samples']
        else:
            return 0

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        """
        Return raw (continuous) signals as 2d numpy array (time x chan).
        Note that block_index and seg_index are always 1 (regardless of input).

        Raw data is in a single vector np.memmap with the following structure:

        Each byte packet (432 bytes) has header (32 bytes), footer (16 bytes)
        and three samples of 2 bytes each for 64 channels (384 bytes), which
        are jumbled up in a strange order. Each channel is remapped to a
        certain position (see get_channel_offset), and a channel's samples are
        allocated as follows (example for channel 7):

        sample 1: 32b (head) + 2*38b (remappedID) and 2*38b + 1b (2nd byte)
        sample 2: 32b (head) + 128 (all chan. 1st entry) + 2*38b and ...
        sample 3: 32b (head) + 128*2 (all channels 1st and 2nd entry) + ...
        """

        bin_dict = self.file_parameters['bin']

        # Set default values
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = bin_dict['num_total_samples']
        if channel_indexes is None:
            channel_indexes = [i for i in range(bin_dict['num_channels'])]
        elif isinstance(channel_indexes, slice):
            channel_indexes_all = [i for i in range(bin_dict['num_channels'])]
            channel_indexes = channel_indexes_all[channel_indexes]

        num_samples = (i_stop - i_start)

        # Create base index vector for _raw_signals for time period of interest
        num_packets_oi = (num_samples + 2) // 3
        offset = i_start // 3 * (bin_dict['bytes_packet'] // 2)
        rem = (i_start % 3)

        raw_samples = np.arange(num_packets_oi + 1, dtype=np.uint32)
        sample1 = raw_samples * (bin_dict['bytes_packet'] // 2) + \
                  bin_dict['bytes_head'] // 2 + offset
        sample2 = sample1 + 64
        sample3 = sample2 + 64

        sig_ids = np.empty((sample1.size + sample2.size + sample3.size,),
                           dtype=sample1.dtype)
        sig_ids[0::3] = sample1
        sig_ids[1::3] = sample2
        sig_ids[2::3] = sample3
        sig_ids = sig_ids[rem:(rem + num_samples)]

        # Read one channel at a time
        raw_signals = np.ndarray(shape=(num_samples,
                                        len(channel_indexes)),
                                 dtype=bin_dict['data_type'])

        for i, ch_idx in enumerate(channel_indexes):
            chan_offset = self.channel_memory_offset[ch_idx]
            raw_signals[:, i] = self._raw_signals[sig_ids + chan_offset]

        return raw_signals

    def _spike_count(self, block_index, seg_index, unit_index):
        tetrode_id = unit_index

        raw_spikes = self._raw_spikes[tetrode_id]
        nb_tetrode_spikes = raw_spikes.shape[0]
        # also take into account last, potentially incomplete set of spikes
        nb_unit_spikes = int(np.ceil(nb_tetrode_spikes))
        return nb_unit_spikes

    def _get_spike_timestamps(self, block_index, seg_index, unit_index,
                              t_start, t_stop):
        assert block_index == 0
        assert seg_index == 0

        tetrode_id = unit_index
        raw_spikes = self._raw_spikes[tetrode_id]

        # spike times are repeated for each contact -> use only first contact
        unit_spikes = raw_spikes['spiketimes'][:, 0]

        # slice spike times only if needed
        if t_start is None and t_stop is None:
            return unit_spikes

        if t_start is None:
            t_start = self._segment_t_start(block_index, seg_index)
        if t_stop is None:
            t_stop = self._segment_t_stop(block_index, seg_index)

        mask = self._get_temporal_mask(t_start, t_stop, tetrode_id)

        return unit_spikes[mask]

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self._to_hz(self.file_parameters['unit']['timebase'],
                                   dtype=int)
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index,
                                 t_start, t_stop):
        assert block_index == 0
        assert seg_index == 0

        tetrode_id = unit_index
        waveforms = self._raw_spikes[tetrode_id]['samples']

        # slice timestamps / waveforms only when necessary
        if t_start is None and t_stop is None:
            return waveforms

        if t_start is None:
            t_start = self._segment_t_start(block_index, seg_index)
        if t_stop is None:
            t_stop = self._segment_t_stop(block_index, seg_index)

        mask = self._get_temporal_mask(t_start, t_stop, tetrode_id)
        waveforms = waveforms[mask]

        return waveforms

    def get_header_bstring(self, file):
        """
        Scan file for the occurrence of 'data_start' and return the header
        as byte string

        Parameters
        ----------
        file (str or path): file to be loaded

        Returns
        -------
        str: header byte content
        """

        header = b''
        with open(file, 'rb') as f:
            for bin_line in f:
                if b'data_start' in bin_line:
                    header += b'data_start'
                    break
                else:
                    header += bin_line
        return header

    # ------------------ HELPER METHODS --------------------
    # This is largely based on code by Geoff Barrett from the Hussaini lab:
    # https://github.com/GeoffBarrett/BinConverter
    # Adapted or modified by Steffen Buergers, Julia Sprenger

    def _get_temporal_mask(self, t_start, t_stop, tetrode_id):
        # Convenience function for creating a temporal mask given
        # start time (t_start) and stop time (t_stop)
        # Used by _get_spike_raw_waveforms and _get_spike_timestamps

        # spike times are repeated for each contact -> use only first contact
        raw_spikes = self._raw_spikes[tetrode_id]
        unit_spikes = raw_spikes['spiketimes'][:, 0]

        # convert t_start and t_stop to sampling frequency
        # Note: this assumes no time offset!
        unit_params = self.file_parameters['unit']
        lim0 = t_start * self._to_hz(unit_params['timebase'], dtype=int)
        lim1 = t_stop * self._to_hz(unit_params['timebase'], dtype=int)

        mask = (unit_spikes >= lim0) & (unit_spikes <= lim1)

        return mask

    def get_header_parameters(self, file, file_type):
        """
        Extract header parameters as dictionary keys and
        following phrases as values (strings).

        Parameters
        ----------
        file (str or path): file to be loaded
        file_type (str): type of file to be loaded ('set' or 'unit')

        Returns
        -------
        header (dict): dictionary with keys being the parameters that
                       were found & values being strings of the data.

        EXAMPLE
        self.get_header_parameters('file.set', 'set')
        """

        params = {}
        encoding = self.file_parameters[file_type]['header_encoding']
        header_string = self.get_header_bstring(file).decode(encoding)

        # omit the last line as this contains only `data_start` key
        for line in header_string.splitlines()[:-1]:
            key, value = line.split(' ', 1)
            params[key] = value
        return params

    def get_active_tetrode(self):
        """
        Returns the ID numbers of the active tetrodes as a list.
        E.g.: [1,2,3,4] for a recording with 4 tetrodes (16 channels).
        """
        active_tetrodes = []

        for key, status in self.file_parameters['set']['file_header'].items():
            # The pattern to look for is collectMask_X Y,
            # where X is the tetrode number, and Y is 1 or 0
            if key.startswith('collectMask_'):
                if int(status):
                    tetrode_id = int(key.strip('collectMask_'))
                    active_tetrodes.append(tetrode_id)
        return active_tetrodes

    def _get_channel_from_tetrode(self, tetrode):
        """
        This function will take the tetrode number and return the Axona
        channel numbers, i.e. Tetrode 1 = Ch0-Ch3, Tetrode 2 = Ch4-Ch7, etc.
        """
        return np.arange(0, 4) + 4 * (int(tetrode) - 1)

    def read_datetime(self):
        """
        Creates datetime object (y, m, d, h, m, s) from .set file header
        """

        date_str = self.file_parameters['set']['file_header']['trial_date']
        time_str = self.file_parameters['set']['file_header']['trial_time']

        # extract core date string
        date_str = re.findall(r'\d+\s\w+\s\d{4}$', date_str)[0]

        return datetime.datetime.strptime(date_str + ', ' + time_str,
                                          "%d %b %Y, %H:%M:%S")

    def _get_channel_gain(self, bytes_per_sample=2):
        """
        This is actually not the gain_ch value from the .set file, but the
        conversion factor from raw data to uV.

        Formula for conversion to uV:

        1000 * adc_fullscale_mv / (gain_ch * max-value), with
        max_value = 2**(8 * bytes_per_sample - 1)

        Adapted from
        https://github.com/CINPLA/pyxona/blob/stable/pyxona/core.py
        """
        gain_list = []

        adc_fm = int(
            self.file_parameters['set']['file_header']['ADC_fullscale_mv'])

        for key, value in self.file_parameters['set']['file_header'].items():
            if key.startswith('gain_ch'):
                gain_list.append(np.float32(value))

        max_value = 2**(8 * bytes_per_sample - 1)
        gain_list = [1000 * adc_fm / (gain * max_value) for gain in gain_list]

        return gain_list

    def _get_signal_chan_header(self):
        """
        Returns a 1 dimensional np.array of tuples with one entry per channel
        that recorded data. Each tuple contains the following information:

        channel name (1a, 1b, 1c, 1d, 2a, 2b, ...; num=tetrode, letter=elec),
        channel id (1, 2, 3, 4, 5, ... N),
        sampling rate,
        data type (int16),
        unit (uV),
        gain,
        offset,
        stream id
        """
        active_tetrode_set = self.get_active_tetrode()
        num_active_tetrode = len(active_tetrode_set)

        elec_per_tetrode = 4
        letters = ['a', 'b', 'c', 'd']
        dtype = self.file_parameters['bin']['data_type']
        units = 'uV'
        gain_list = self._get_channel_gain()
        offset = 0  # What is the offset?

        sig_channels = []
        for itetr in range(num_active_tetrode):

            for ielec in range(elec_per_tetrode):
                cntr = (itetr * elec_per_tetrode) + ielec
                ch_name = '{}{}'.format(itetr + 1, letters[ielec])
                chan_id = str(cntr)
                gain = gain_list[cntr]
                stream_id = '0'
                # the sampling rate information is stored in the set header
                # and not in the bin file
                sr = self.file_parameters['set']['sampling_rate']
                sig_channels.append((ch_name, chan_id, sr, dtype,
                                     units, gain, offset, stream_id))

        return np.array(sig_channels, dtype=_signal_channel_dtype)

    def _to_hz(self, param, dtype=float):
        return dtype(param.replace(' hz', ''))
