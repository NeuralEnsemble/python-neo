"""

File format overview:
http://space-memory-navigation.org/DacqUSBFileFormats.pdf

In brief:
 data.set setup file containing all hardware setups related to the trial
 data.bin raw data file
 data.1 spike times and waveforms for tetrode 1, or stereotrodes 1 and 2
 data.2 spikes times and waveforms for tetrode 2, or stereotrodes 3 and 4
 …
 data.32 spikes times and waveforms for tetrode 32
 data.eeg continuous 250 Hz EEG signal, primary channel
 data.eegX continuous 250 Hz EEG signal, secondary channel (X = 1..16)
 data.egf high resolution 4800 Hz version of primary EEG channel
 data.egfX high resolution 4800 Hz version of primary EEG channel (X = 1..16)
 data.pos tracker position data
 data.inp digital input and keypress timestamps
 data.stm stimulation pulse timestamps
 data.spk spikes times and waveforms for monotrodes (single electrodes) 1 to 16
 data.epp field potential parameters
 data.epw field potential waveforms
 data.log DACQBASIC script optional user-defined output files

Here we are only going to support .set and .bin files for now (raw cont. data). 
 At least the following could be derived from it: .eeg, .egf, .X, .pos.

Rules for creating a new class:
  1. Step 1: Create the main class

  2. Step 2: RawIO test:

  3. Step 3 : Create the neo.io class with the wrapper
    * Create a file in neo/io/ that ends with "io.py"
    * Create a class that inherits both your RawIO class and BaseFromRaw class
    * copy/paste from neo/io/exampleio.py

  4.Step 4 : IO test
    * create a file in neo/test/iotest with the same previous name with "test_" prefix
    * copy/paste from neo/test/iotest/test_exampleio.py

Author: Steffen Buergers

"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype, _common_sig_characteristics)
import numpy as np
import os
import re
import contextlib
import datetime


class AxonaRawIO(BaseRawIO):
    """
    Class for reading raw, continuous data from the Axona dacqUSB system:
    http://space-memory-navigation.org/DacqUSBFileFormats.pdf

    The raw data is saved in .bin binary files with an accompanying
    .set file about the recording setup (see the above manual for details).

    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.AxonaRawIO(filename=os.path.join(dir_name, base_filename))
        >>> r.parse_header()
        >>> print(r)
        >>> raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=0, i_stop=1024,  channel_names=channel_names)
        >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, dtype='float64',
                            channel_indexes=[0, 3, 6])
        >>> spike_timestamp = reader.spike_timestamps(spike_channel_index=0,
                            t_start=None, t_stop=None)
        >>> spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
        >>> ev_timestamps, _, ev_labels = reader.event_timestamps(event_channel_index=0)

    """
    # TODO Why do I need these?
    extensions = ['bin']
    rawmode = 'one-file'

    def __init__(self, filename='', sr=48000):
        BaseRawIO.__init__(self)

        # note that this filename is used in self._source_name
        self.filename = filename
        self.bin_file = os.path.join(self.filename + '.bin') 
        self.set_file = os.path.join(self.filename + '.set')
        self.set_file_encoding = 'cp1252'

        # Useful num. bytes per continuous data packet (.bin file)
        self.bytes_packet = 432
        self.bytes_data = 384
        self.bytes_head = 32
        self.bytes_tail = 16

        # There is no global header for .bin files
        self.global_header_size = 0

        self.sr = sr
        self.num_channels = len(self.get_active_tetrode()) * 4

        # How many 432 byte packets does this data contain (<=> num. samples / 3)?
        self.num_total_packets = int(os.path.getsize(self.bin_file)/self.bytes_packet)
        self.num_total_samples = self.num_total_packets * 3

    def _source_name(self):
        # this function is used by __repr__
        # for general cases self.filename is good
        # But for URL you could mask some part of the URL to keep
        # the main part.
        return self.filename

    def _parse_header(self):
        # This is the central of a RawIO
        # we need to collect in the original format all
        # informations needed for further fast acces
        # at any place in the file
        # In short _parse_header can be slow but
        # _get_analogsignal_chunk need to be as fast as possible

        # Create np.memmap to .bin file
        self._raw_signals = np.memmap(self.bin_file, dtype='int16', mode='r', 
                                      offset=self.global_header_size)

        # fill into header dict
        # This is mandatory!!!!!

        

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = self.get_signal_streams_header()
        self.header['signal_channels'] = self.get_signal_chan_header()
        self.header['spike_channels'] = self.get_spike_chan_header()
        self.header['event_channels'] = self.get_event_chan_header()

        # insert some annotation at some place
        # at neo.io level IO are free to add some annoations
        # to any object. To keep this functionality with the wrapper
        # BaseFromRaw you can add annoations in a nested dict.
        self._generate_minimal_annotations()

    def get_signal_streams_header(self):
        '''
        create signals stream information (we always expect a single stream)
        '''
        return np.array([('stream 0', 0)], dtype=_signal_stream_dtype)

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        return self.num_total_samples / self.sr

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        return self.num_total_samples

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        """
        Return raw (continuous) signals as 2d numpy array (time x chan).
        Note that block_index and seg_index are always 1 (regardless of input).
        
        The raw data is in a single vector np.memmap with the following structure:
        
        Each byte packet (432 bytes) has a header (32 bytes), a footer (16 bytes)
        and three samples of 2 bytes each for 64 channels (384 bytes), which are 
        jumbled up in a strange order. Each channel is remapped to a certain position
        (see get_channel_offset), and a channel's samples are allcoated as follows
        (example for channel 7):
        
        sample 1: 32b (head) + 2*38b (remappedID) and 2*38b + 1b (second byte of sample)
        sample 2: 32b (head) + 128 (all channels 1st entry) + 2*38b (remappedID) and ...
        sample 3: 32b (head) + 128*2 (all channels 1st and 2nd entry) + ...

        NOTE: I believe there is always a single stream (all channels have the same SR)
        """            

        if channel_indexes is None:
            channel_indexes = [i+1 for i in range(self.num_channels)]

        num_samples = (i_stop-i_start)

        # Create base index vector for _raw_signals for time period of interest
        num_packets_oi = (num_samples+2) // 3
        offset = i_start//3 * (self.bytes_packet//2) 
        rem = (i_start % 3)

        sample1 = np.arange(num_packets_oi+1, dtype=np.uint32)*(self.bytes_packet//2) + \
                    self.bytes_head//2 + offset
        sample2 = sample1 + 64
        sample3 = sample2 + 64

        sig_ids = np.empty((sample1.size+sample2.size+sample3.size,), dtype=sample1.dtype)
        sig_ids[0::3] = sample1
        sig_ids[1::3] = sample2
        sig_ids[2::3] = sample3
        sig_ids = sig_ids[rem:(rem+num_samples)]

        # Read one channel at a time
        raw_signals = np.ndarray(shape=(num_samples, len(channel_indexes)))

        for i, ch_idx in enumerate(channel_indexes):

            chan_offset = self.get_channel_offset(ch_idx)
            raw_signals[:,i] = self._raw_signals[sig_ids + chan_offset]

        return raw_signals

    def _spike_count(self, block_index, seg_index, unit_index):
        return None

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        return None

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        return None

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):
        return None

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        return None

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        return None

    def _rescale_epoch_duration(self, raw_duration, dtype):
        return None

    # ------------------ HELPER METHODS --------------------
    # These are credited largely to Geoff Barrett from the Hussaini lab:
    # https://github.com/GeoffBarrett/BinConverter
    # Adapted or modified by Steffen Buergers

    def get_active_tetrode(self):
        """ 
        Returns the ID numbers of the active tetrodes (those with recorded data)
        as a list. E.g.: [1,2,3,4] for a recording with 4 tetrodes (16 channels).
        """
        active_tetrodes = []

        with open(self.set_file, encoding=self.set_file_encoding) as f:
            for line in f:

                # The pattern to look for is collectMask_X Y, 
                # where X is the tetrode number, and Y is true or false (1, 0)
                if 'collectMask_' in line:
                    tetrode_str, tetrode_status = line.split(' ')
                    if int(tetrode_status) == 1:
                        tetrode_id = int(re.findall(r'\d+', tetrode_str)[0])
                        active_tetrodes.append(tetrode_id)

        return active_tetrodes

    def get_channel_from_tetrode(self, tetrode):
        """ 
        This function will take the tetrode number and return the Axona channel numbers
        i.e. Tetrode 1 = Ch1-Ch4, Tetrode 2 = Ch5-Ch8, etc.
        """
        return np.arange(1, 5) + 4 * (int(tetrode) - 1)

    def get_channel_offset(self, chan_id):
        """ 
        In the .bin file, channels are arranged in a strange order. This method takes
        a channel index as input and returns the actual offset for the channel in the
        memory map (self._raw_signals).
        """
        channnel_memory_offset = np.array(
            [32, 33, 34, 35, 36, 37, 38, 39, 0, 1, 2, 3, 4, 5,
             6, 7, 40, 41, 42, 43, 44, 45, 46, 47, 8, 9, 10, 11,
             12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 16, 17,
             18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63,
             24, 25, 26, 27, 28, 29, 30, 31]
        )
        return channnel_memory_offset[chan_id - 1]

    def read_datetime(self):
        """ 
        Creates datetime object (y, m, d, h, m, s) from .set file header 
        """
        with open(self.set_file, 'r', encoding=self.set_file_encoding) as f:
            for line in f:
                if line.startswith('trial_date'):
                    date_string = re.findall(r'\d+\s\w+\s\d{4}$', line)[0]
                if line.startswith('trial_time'):
                    time_string = line[len('trial_time')+1::].replace('\n', '')

        return datetime.datetime.strptime(date_string + ', ' + time_string, \
            "%d %b %Y, %H:%M:%S")

    def get_channel_gain(self):
        """ 
        Read gain for each channel from .set file and return in list of integers 

        This is actually not the gain_ch value from the .set file, but the conversion
        factor from raw data to uV.

        Formula for .eeg and .X files, presumably also .bin files:    
        
        1000*adc_fullscale_mv / (gain_ch*128)
        """
        gain_list = []

        with open(self.set_file, encoding='cp1252') as f:
            for line in f:
                if line.startswith('ADC_fullscale_mv'):
                    adc_fullscale_mv = int(line.split(" ")[1])
                if line.startswith('gain_ch'):
                    gain_list.append(np.float32(re.findall(r'\d*', line.split(' ')[1])[0]))
        
        return [1000*adc_fullscale_mv/(gain*128) for gain in gain_list]

    def get_signal_chan_header(self):
        """
        Returns a 1 dimensional np.array of tuples with one entry for each channel
        that recorded data. Each tuple contains the following information:

        channel name (1a, 1b, 1c, 1d, 2a, 2b, ...; with num=tetrode, letter=electrode), 
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
        dtype = 'int16'
        units = 'uV'
        gain_list = self.get_channel_gain()
        offset = 0  # What is the offset? 

        sig_channels = []
        for itetr in range(num_active_tetrode):

            for ielec in range(elec_per_tetrode):
                
                cntr = (itetr*elec_per_tetrode) + ielec
                ch_name = '{}{}'.format(itetr, letters[ielec])
                chan_id = cntr + 1
                gain = gain_list[cntr]
                stream_id = 0
                sig_channels.append((ch_name, chan_id, self.sr, dtype, 
                    units, gain, offset, stream_id))
                
        return np.array(sig_channels, dtype=_signal_channel_dtype) 

    def get_spike_chan_header(self):
        """
        No spikes currently
        """
        return  np.array([], dtype=_spike_channel_dtype)

    def get_event_chan_header(self):
        """
        No events currently
        """
        return np.array([], dtype=_event_channel_dtype)

# eof

