"""
This class reads .set and .bin file data from the Axona acquisition system.

File format overview:
http://space-memory-navigation.org/DacqUSBFileFormats.pdf

In brief:
 data.set - setup file containing all hardware setups related to the trial
 data.bin - raw data file

There are many other data formats from Axona, which we do not consider (yet).
These are derivatives from the raw continuous data (.bin) and could in
principle be extracted from it (see file format overview for details).

Every recording contains extracellular electrophysiology (ecephy) data in
stream 0. When available video tracking (pos) data is in stream 1.

Author: Steffen Buergers

"""

from neo.rawio.baserawio import (
    BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
    _spike_channel_dtype, _event_channel_dtype
)
import numpy as np
import os
import re
import datetime
import contextlib
import mmap


class AxonaRawIO(BaseRawIO):
    """
    Class for reading raw, continuous data from the Axona dacqUSB system:
    http://space-memory-navigation.org/DacqUSBFileFormats.pdf

    The raw data is saved in .bin binary files with an accompanying
    .set file about the recording setup (see the above manual for details).

    Usage:
        import neo

        r = neo.rawio.AxonaRawIO(
            filename=os.path.join(dir_name, set_filename)
        )
        r.parse_header()
        print('Header information:\n', r)

        raw_chunk = r.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=5,
            stream_index=0, channel_indexes=[0, 3, 6]
        )
        print('Raw acquisition traces:\n', raw_chunk)

        float_chunk = r.rescale_signal_raw_to_float(
            raw_chunk, dtype=np.float64,
            channel_indexes=[0, 3, 6],
            stream_index=0
        )
        print('\nRaw acquisition traces in uV:\n', float_chunk)
    """

    extensions = ['bin']  # Never used?
    rawmode = 'one-file'

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

        # We accept base filenames, .bin and .set extensions
        self.filename = filename.replace('.bin', '').replace('.set', '')
        self.bin_file = os.path.join(self.filename + '.bin')
        self.set_file = os.path.join(self.filename + '.set')
        self.set_file_encoding = 'cp1252'

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        '''
        Read important information from .set header file, create memory map
        to raw data (.bin file) and prepare header dictionary in neo format.
        '''

        # Ecephys
        params = ['rawRate']
        params = self.get_set_file_parameters(params)

        self.bytes_packet = 432
        self.bytes_data = 384
        self.bytes_head = 32
        self.bytes_tail = 16

        self.data_dtype = np.int16

        self.global_header_size = 0

        self.sr_ecephys = int(params['rawRate'])
        self.num_channels_ecephys = len(self.get_active_tetrode()) * 4

        self.num_total_packets = int(os.path.getsize(self.bin_file)
                                     / self.bytes_packet)
        self.num_ecephys_samples = self.num_total_packets * 3
        self.dur_ecephys = self.num_ecephys_samples / self.sr_ecephys

        self._raw_signals = np.memmap(
            self.bin_file, dtype=self.data_dtype, mode='r',
            offset=self.global_header_size
        )

        # Pos (video) tracking
        self.sr_pos = 100  # ideal SR, empirical SR might differ
        with open(self.bin_file, 'rb') as f:
            with contextlib.closing(
                mmap.mmap(f.fileno(), self.sr_ecephys // 3 // self.sr_pos
                          * self.bytes_packet, access=mmap.ACCESS_READ)
            ) as mmap_obj:
                self.contains_pos_tracking = mmap_obj.find(b'ADU2') > -1

        if self.contains_pos_tracking:
            self.num_channels_pos = 8
            self.dur_pos = self.dur_ecephys
            self.num_pos_samples = int(self.dur_pos * self.sr_pos)

            fbin = open(self.bin_file, 'rb')
            self.mmpos = mmap.mmap(fbin.fileno(), 0, access=mmap.ACCESS_READ)

        # Header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = self._get_signal_streams_header()
        self.header['signal_channels'] = self._get_signal_chan_header()
        self.header['spike_channels'] = np.array([],
                                                 dtype=_spike_channel_dtype)
        self.header['event_channels'] = np.array([],
                                                 dtype=_event_channel_dtype)

        # Annotations
        self._generate_minimal_annotations()

        bl_ann = self.raw_annotations['blocks'][0]
        seg_ann = bl_ann['segments'][0]
        seg_ann['rec_datetime'] = self.read_datetime()
        sig_an = \
            seg_ann['signals'][0]['__array_annotations__']['tetrode_id'] = \
            [tetr for tetr in self.get_active_tetrode() for _ in range(4)]

    def _get_signal_streams_header(self):
        if self.contains_pos_tracking:
            return np.array([('stream 0', '0'), ('stream 1', '1')],
                            dtype=_signal_stream_dtype)
        return np.array([('stream 0', '0')], dtype=_signal_stream_dtype)

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        return self.num_ecephys_samples / self.sr_ecephys

    def _get_signal_size(self, block_index, seg_index, stream_index):
        if stream_index == 0:
            return self.num_ecephys_samples
        elif (stream_index == 1) and (self.contains_pos_tracking):
            return self.num_pos_samples
        return None

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        """
        Return raw (continuous) signals as 2d numpy array for ecephys
        data (stream 0; time x chan) or video tracking (stream 1; time x 9)
        """
        if stream_index == 1:
            if self.contains_pos_tracking:
                raw_signals = self._get_pos_analogsignal_chunk(
                    i_start, i_stop, channel_indexes
                )

        elif stream_index == 0:
            raw_signals = self._get_ecephys_analogsignal_chunk(
                i_start, i_stop, channel_indexes
            )
        return raw_signals

    # ------------------ HELPER METHODS --------------------
    # These are credited largely to Geoff Barrett from the Hussaini lab:
    # https://github.com/GeoffBarrett/BinConverter
    # Adapted or modified by Steffen Buergers

    def _get_ecephys_analogsignal_chunk(self, i_start, i_stop, channel_indexes):
        """
        Return raw (continuous) signals as 2d numpy array (time x chan).

        Raw data is in a single vector np.memmap with the following structure:

        Each byte packet (432 bytes) has header (32 bytes), footer (16 bytes)
        and three samples of 2 bytes each for 64 channels (384 bytes), which
        are jumbled up in a strange order. Each channel is remapped to a
        certain position (see get_channel_offset), and a channel's samples are
        allcoated as follows (example for channel 7):

        sample 1: 32b (head) + 2*38b (remappedID) and 2*38b + 1b (2nd byte)
        sample 2: 32b (head) + 128 (all chan. 1st entry) + 2*38b and ...
        sample 3: 32b (head) + 128*2 (all channels 1st and 2nd entry) + ...
        """

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.num_ecephys_samples
        if channel_indexes is None:
            channel_indexes = [i for i in range(self.num_channels_ecephys)]

        num_samples = (i_stop - i_start)

        # Create base index vector for _raw_signals for time period of interest
        num_packets_oi = (num_samples + 2) // 3
        offset = i_start // 3 * (self.bytes_packet // 2)
        rem = (i_start % 3)

        sample1 = np.arange(num_packets_oi + 1, dtype=np.uint32) * \
            (self.bytes_packet // 2) + self.bytes_head // 2 + offset
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
                                 dtype=self.data_dtype)

        for i, ch_idx in enumerate(channel_indexes):

            chan_offset = self.channel_memory_offset[ch_idx]
            raw_signals[:, i] = self._raw_signals[sig_ids + chan_offset]

        return raw_signals

    def _get_pos_analogsignal_chunk(self, i_start=0, i_stop=None,
                                    channel_indexes=None):
        """
        Return np.array of video position tracking (Nsamp x 8 columns),
        with columns being the following:

        timestamp, x1, y1, x2, y2, npix1, npix2, total_pix, unused

        Position samples are saved in the header of packets with the
        flag 'ADU2'.
        """
        if channel_indexes is None:
            channel_indexes = [i for i in range(self.num_channels_pos)]

        pos = np.array([]).astype(float)

        flags = np.ndarray(
            (self.num_total_packets,), 'S4',
            self.mmpos, 0, self.bytes_packet
        )
        ADU2_idx = np.where(flags == b'ADU2')

        pos = np.ndarray(
            (self.num_total_packets,), (np.int16, (1, 8)), self.mmpos, 16,
            (self.bytes_packet,)
        ).reshape((-1, 8))[ADU2_idx][:]

        pos = np.hstack(
            (ADU2_idx[0].reshape((-1, 1)), pos)
        ).astype(float)

        # The timestamp from the recording is dubious, create our own
        packets_per_ms = self.sr_ecephys / 3000
        pos[:, 0] = pos[:, 0] / packets_per_ms
        pos = np.delete(pos, 1, 1)

        return pos[i_start:i_stop, channel_indexes]

    def get_set_file_parameters(self, params):
        """
        Given a list of param., looks for each in first word of a phrase
        in the .set file. Adds found paramters as dictionary keys and
        following phrases as values (strings).

        INPUT
        params (list or set): parameter names to search for

        OUTPUT
        header (dict): dictionary with keys being the parameters that
                       were found & values being strings of the data.

        EXAMPLE
        self.get_set_file_parameters(['experimenter', 'trial_time'])
        """
        header = {}
        params = set(params)
        with open(self.set_file, 'rb') as f:
            for bin_line in f:
                if b'data_start' in bin_line:
                    break
                line = bin_line.decode('cp1252').replace('\r\n', '').\
                    replace('\r', '').strip()
                parts = line.split(' ')
                key = parts[0]
                if key in params:
                    header[key] = ' '.join(parts[1:])

        return header

    def get_active_tetrode(self):
        """
        Returns the ID numbers of the active tetrodes as a list.
        E.g.: [1,2,3,4] for a recording with 4 tetrodes (16 channels).
        """
        active_tetrodes = []

        with open(self.set_file, encoding=self.set_file_encoding) as f:
            for line in f:

                # The pattern to look for is collectMask_X Y,
                # where X is the tetrode number, and Y is 1 or 0
                if 'collectMask_' in line:
                    tetrode_str, tetrode_status = line.split(' ')
                    if int(tetrode_status) == 1:
                        tetrode_id = int(re.findall(r'\d+', tetrode_str)[0])
                        active_tetrodes.append(tetrode_id)

        return active_tetrodes

    def _get_channel_from_tetrode(self, tetrode):
        """
        This function will take the tetrode number and return the Axona
        channel numbers, i.e. Tetrode 1 = Ch1-Ch4, Tetrode 2 = Ch5-Ch8, etc.
        """
        return np.arange(1, 5) + 4 * (int(tetrode) - 1)

    def read_datetime(self):
        """
        Creates datetime object (y, m, d, h, m, s) from .set file header
        """
        with open(self.set_file, 'r', encoding=self.set_file_encoding) as f:
            for line in f:
                if line.startswith('trial_date'):
                    date_string = re.findall(r'\d+\s\w+\s\d{4}$', line)[0]
                if line.startswith('trial_time'):
                    time_string = line[len('trial_time') + 1::].replace('\n', '')

        return datetime.datetime.strptime(date_string + ', ' + time_string,
                                          "%d %b %Y, %H:%M:%S")

    def _get_channel_gain(self):
        """
        Read gain for each channel from .set file and return list of integers

        This is actually not the gain_ch value from the .set file, but the
        conversion factor from raw data to uV.

        Formula for .eeg and .X files, presumably also .bin files:

        1000 * adc_fullscale_mv / (gain_ch*128)
        """
        gain_list = []

        with open(self.set_file, encoding='cp1252') as f:
            for line in f:
                if line.startswith('ADC_fullscale_mv'):
                    adc_fullscale_mv = int(line.split(" ")[1])
                if line.startswith('gain_ch'):
                    gain_list.append(
                        np.float32(re.findall(r'\d*', line.split(' ')[1])[0])
                    )

        return [1000 * adc_fullscale_mv / (gain * 128) for gain in gain_list]

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

        When available, video tracking data is appended after the ecephys data.
        """

        # Ecephys data
        active_tetrode_set = self.get_active_tetrode()
        num_active_tetrode = len(active_tetrode_set)

        elec_per_tetrode = 4
        letters = ['a', 'b', 'c', 'd']
        dtype = self.data_dtype
        units = 'uV'
        gain_list = self._get_channel_gain()
        offset = 0

        sig_channels = []
        for itetr in range(num_active_tetrode):

            for ielec in range(elec_per_tetrode):

                cntr = (itetr * elec_per_tetrode) + ielec
                ch_name = '{}{}'.format(itetr + 1, letters[ielec])
                chan_id = str(cntr)
                gain = gain_list[cntr]
                stream_id = '0'
                sig_channels.append((ch_name, chan_id, self.sr_ecephys, dtype,
                                     units, gain, offset, stream_id))

        # Append video tracking data
        if self.contains_pos_tracking:

            # NOTE: In the file format Manual there are two recording modes
            # for video tracking: Four-spot mode and two-spot mode. Here we
            # only consider two-spot mode!
            pos_chan_names = 't,x1,y1,x2,y2,numpix1,numpix2,unused'.split(',')
            pos_units = ['', '', '', '', '', 'pix', 'pix', '']
            for i, (name, unit) in enumerate(zip(pos_chan_names, pos_units)):
                sig_channels.append(
                    (name, str(i), self.sr_pos, np.float64,
                     unit, 1, 0, '1')
                )

        return np.array(sig_channels, dtype=_signal_channel_dtype)
