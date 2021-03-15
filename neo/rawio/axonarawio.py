"""

File format overview:
http://space-memory-navigation.org/DacqUSBFileFormats.pdf

In brief:
 data.set setup file containing all hardware setups related to the trial
 data.1 spike times and waveforms for tetrode 1, or stereotrodes 1 and 2
 data.2 spikes times and waveforms for tetrode 2, or stereotrodes 3 and 4
 …
 data.32 spikes times and waveforms for tetrode 32
 data.spk spikes times and waveforms for monotrodes (single electrodes) 1 to 16
 data.eeg continuous 250 Hz EEG signal, primary channel
 data.eegX continuous 250 Hz EEG signal, secondary channel (X = 1..16)
 data.egf high resolution 4800 Hz version of primary EEG channel
 data.egfX high resolution 4800 Hz version of primary EEG channel (X = 1..16)
 data.pos tracker position data
 data.inp digital input and keypress timestamps
 data.stm stimulation pulse timestamps
 data.bin raw data file
 data.epp field potential parameters
 data.epw field potential waveforms
 data.log DACQBASIC script optional user-defined output files 

ExampleRawIO is a class of a  fake example.
This is to be used when coding a new RawIO.


Rules for creating a new class:
  1. Step 1: Create the main class
    * Create a file in **neo/rawio/** that endith with "rawio.py"
    * Create the class that inherits BaseRawIO
    * copy/paste all methods that need to be implemented.
      See the end a neo.rawio.baserawio.BaseRawIO
    * code hard! The main difficulty **is _parse_header()**.
      In short you have a create a mandatory dict than
      contains channel informations::

            self.header = {}
            self.header['nb_block'] = 2
            self.header['nb_segment'] = [2, 3]
            self.header['signal_channels'] = sig_channels
            self.header['unit_channels'] = unit_channels
            self.header['event_channels'] = event_channels

  2. Step 2: RawIO test:
    * create a file in neo/rawio/tests with the same name with "test_" prefix
    * copy paste neo/rawio/tests/test_examplerawio.py and do the same

  3. Step 3 : Create the neo.io class with the wrapper
    * Create a file in neo/io/ that endith with "io.py"
    * Create a that inherits both your RawIO class and BaseFromRaw class
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
    Class for reading raw data from the Axona dacqUSB system:
    http://space-memory-navigation.org/DacqUSBFileFormats.pdf

    The raw data is save in .bin binary files. 

    For the user, it give acces to raw data (signals, event, spikes) as they
    are in the (fake) file int16 and int64.

    For a developer, it is just an example showing guidelines for someone who wants
    to develop a new IO module.

    Two rules for developers:
      * Respect the :ref:`neo_rawio_API`
      * Follow the :ref:`io_guiline`

    This fake IO:
        * have 2 blocks
        * blocks have 2 and 3 segments
        * have 16 signal_channel sample_rate = 10000
        * have 3 unit_channel
        * have 2 event channel: one have *type=event*, the other have
          *type=epoch*


    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.ExampleRawIO(filename='itisafake.nof')
        >>> r.parse_header()
        >>> print(r)
        >>> raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=0, i_stop=1024,  channel_names=channel_names)
        >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, dtype='float64',
                            channel_indexes=[0, 3, 6])
        >>> spike_timestamp = reader.spike_timestamps(unit_index=0, t_start=None, t_stop=None)
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

        # How many 432 byte packets does this data contain (<=> num. samples / 3)?
        self.num_packets = int(os.path.getsize(self.bin_file)/self.bytes_packet)

        # Create np.memmap to .bin file
        self._raw_signals = np.memmap(self.bin_file, dtype='int16', mode='r', 
                                      offset=self.global_header_size)
        
        # Create base index vector for _raw_signals
        sample1 = np.linspace(self.bytes_head//2, self.num_packets*(self.bytes_packet//2), 
                              self.num_packets, dtype=int)
        sample2 = sample1 + 64
        sample3 = sample2 + 64

        self.sig_ids = np.empty((sample1.size+sample2.size+sample3.size,), dtype=sample1.dtype)
        self.sig_ids[0::3] = sample1
        self.sig_ids[1::3] = sample2
        self.sig_ids[2::3] = sample3

        # fille into header dict
        # This is mandatory!!!!!

        # create fake signals stream information
        signal_streams = []
        for c in range(1):
            name = f'stream {c}'
            stream_id = c
            signal_streams.append((name, stream_id))
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = self.get_signal_chan_header()
        self.header['spike_channels'] = self.get_spike_chan_header()
        self.header['event_channels'] = self.get_event_chan_header()

        # insert some annotation at some place
        # at neo.io level IO are free to add some annoations
        # to any object. To keep this functionality with the wrapper
        # BaseFromRaw you can add annoations in a nested dict.
        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        # this must return an float scale in second
        # this t_start will be shared by all object in the segment
        # except AnalogSignal
        all_starts = [[0., 15.], [0., 20., 60.]]
        return all_starts[block_index][seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        # this must return an float scale in second
        all_stops = [[10., 25.], [10., 30., 70.]]
        return all_stops[block_index][seg_index]

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        # we are lucky: signals in all segment have the same shape!! (10.0 seconds)
        # it is not always the case
        # this must return an int = the number of sample

        # Note that channel_indexes can be ignored for most cases
        # except for several sampling rate.
        return 100000

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        # This give the t_start of signals.
        # Very often this equal to _segment_t_start but not
        # always.
        # this must return an float scale in second

        # Note that channel_indexes can be ignored for most cases
        # except for several sampling rate.

        # Here this is the same.
        # this is not always the case
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        """
        Return raw (continuous) signals as 2d numpy array (chan x time).
        Note that block_index and seg_index are always 1 (regardless of input).
        
        The raw data is in a single vector np.memmap with the following structure:
        
        Each byte packet (432 bytes) has a header (32 bytes), a footer (16 bytes)
        and three samples of 2 bytes each for 64 channels (384 bytes), which are 
        jumbled up in a strange order. Each channel is remapped to a certain position
        (see get_remap_chan), and a channel's samples are allcoated as follows
        (example for channel 7):
        
        sample 1: 32b (head) + 2*38b (remappedID) and 2*38b + 1b (second byte of sample)
        sample 2: 32b (head) + 128 (all channels 1st entry) + 2*38b (remappedID) and ...
        sample 3: 32b (head) + 128*2 (all channels 1st and 2nd entry) + ...

        NOTE: I believe there is always a single stream (all channels have the same SR)
        """
        
        if channel_indexes is None:
            channel_indexes = [i+1 for i in range(self.num_channels)]

        # Each packet has three samples for each channel, so for user_offset
        # we move i_start//3 packets + whatever remainder is left * 64
        num_samples = (i_stop-i_start)

        # Read one channel at a time
        raw_signals = np.ndarray(shape=(len(channel_indexes), num_samples))

        for i, ch_idx in enumerate(channel_indexes):

            chan_offset = self.get_remap_chan(ch_idx)
            raw_signals[i,:] = self._raw_signals[self.sig_ids[i_start:i_stop] + chan_offset]
        
        return raw_signals

    def _spike_count(self, block_index, seg_index, unit_index):
        # Must return the nb of spike for given (block_index, seg_index, unit_index)
        # we are lucky:  our units have all the same nb of spikes!!
        # it is not always the case
        nb_spikes = 20
        return nb_spikes

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        # In our IO, timstamp are internally coded 'int64' and they
        # represent the index of the signals 10kHz
        # we are lucky: spikes have the same discharge in all segments!!
        # incredible neuron!! This is not always the case

        # the same clip t_start/t_start must be used in _spike_raw_waveforms()

        ts_start = (self._segment_t_start(block_index, seg_index) * 10000)

        spike_timestamps = np.arange(0, 10000, 500) + ts_start

        if t_start is not None or t_stop is not None:
            # restricte spikes to given limits (in seconds)
            lim0 = int(t_start * 10000)
            lim1 = int(t_stop * 10000)
            mask = (spike_timestamps >= lim0) & (spike_timestamps <= lim1)
            spike_timestamps = spike_timestamps[mask]

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        # must rescale to second a particular spike_timestamps
        # with a fixed dtype so the user can choose the precisino he want.
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= 10000.  # because 10kHz
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        # this must return a 3D numpy array (nb_spike, nb_channel, nb_sample)
        # in the original dtype
        # this must be as fast as possible.
        # the same clip t_start/t_start must be used in _spike_timestamps()

        # If there there is no waveform supported in the
        # IO them _spike_raw_waveforms must return None

        # In our IO waveforms come from all channels
        # they are int16
        # convertion to real units is done with self.header['unit_channels']
        # Here, we have a realistic case: all waveforms are only noise.
        # it is not always the case
        # we 20 spikes with a sweep of 50 (5ms)

        # trick to get how many spike in the slice
        ts = self._get_spike_timestamps(block_index, seg_index, unit_index, t_start, t_stop)
        nb_spike = ts.size

        np.random.seed(2205)  # a magic number (my birthday)
        waveforms = np.random.randint(low=-2**4, high=2**4, size=nb_spike * 50, dtype='int16')
        waveforms = waveforms.reshape(nb_spike, 1, 50)
        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        # event and spike are very similar
        # we have 2 event channels
        if event_channel_index == 0:
            # event channel
            return 6
        elif event_channel_index == 1:
            # epoch channel
            return 10

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        # the main difference between spike channel and event channel
        # is that for here we have 3 numpy array timestamp, durations, labels
        # durations must be None for 'event'
        # label must a dtype ='U'

        # in our IO event are directly coded in seconds
        seg_t_start = self._segment_t_start(block_index, seg_index)
        if event_channel_index == 0:
            timestamp = np.arange(0, 6, dtype='float64') + seg_t_start
            durations = None
            labels = np.array(['trigger_a', 'trigger_b'] * 3, dtype='U12')
        elif event_channel_index == 1:
            timestamp = np.arange(0, 10, dtype='float64') + .5 + seg_t_start
            durations = np.ones((10), dtype='float64') * .25
            labels = np.array(['zoneX'] * 5 + ['zoneZ'] * 5, dtype='U12')

        if t_start is not None:
            keep = timestamp >= t_start
            timestamp, labels = timestamp[keep], labels[keep]
            if durations is not None:
                durations = durations[keep]

        if t_stop is not None:
            keep = timestamp <= t_stop
            timestamp, labels = timestamp[keep], labels[keep]
            if durations is not None:
                durations = durations[keep]

        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        # must rescale to second a particular event_timestamps
        # with a fixed dtype so the user can choose the precisino he want.

        # really easy here because in our case it is already seconds
        event_times = event_timestamps.astype(dtype)
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype):
        # really easy here because in our case it is already seconds
        durations = raw_duration.astype(dtype)
        return durations

    # ------------------ HELPER METHODS --------------------
    # These are credited to Geoff Barrett from the Hussaini lab:
    # https://github.com/GeoffBarrett/BinConverter
    # Possibly modified by Steffen B
    def get_active_tetrode(self):
        """ 
        In the .set files it will say collectMask_X Y for each tetrode number to tell 
        you if it is active or not. T1 = ch1-ch4, T2 = ch5-ch8, etc.
        """
        active_tetrode = []
        active_tetrode_str = 'collectMask_'

        with open(self.set_file, encoding=self.set_file_encoding) as f:
            for line in f:

                # collectMask_X Y, where x is the tetrode number, and Y is eitehr on or off (1 or 0)
                if active_tetrode_str in line:
                    tetrode_str, tetrode_status = line.split(' ')
                    if int(tetrode_status) == 1:
                        # then the tetrode is saved
                        tetrode_str.find('_')
                        tet_number = int(tetrode_str[tetrode_str.find('_') + 1:])
                        active_tetrode.append(tet_number)

        return active_tetrode

    def get_channel_from_tetrode(self, tetrode):
        """ 
        This function will take the tetrode number and return the Axona channel numbers
        i.e. Tetrode 1 = Ch1 -Ch4, Tetrode 2 = Ch5-Ch8, etc
        """
        tetrode = int(tetrode)  # just in case the user gave a string as the tetrode

        return np.arange(1, 5) + 4 * (tetrode - 1)

    def get_sample_indices(self, channel_number, samples):
        remap_channel = self.get_remap_chan(channel_number)

        indices_scalar = np.multiply(np.arange(samples), 64)
        sample_indices = indices_scalar + np.multiply(np.ones(samples), remap_channel)

        # return np.array([remap_channel, 64 + remap_channel, 64*2 + remap_channel])
        return (indices_scalar + np.multiply(np.ones(samples), remap_channel)).astype(int)

    def get_remap_chan(self, chan_num):
        """ 
        There is re-mapping, thus to get the correct channel data, you need to 
        incorporate re-mapping input will be a channel from 1 to 64, and will return 
        the remapped channel.
        """

        remap_channels = np.array([32, 33, 34, 35, 36, 37, 38, 39, 0, 1, 2, 3, 4, 5,
                                6, 7, 40, 41, 42, 43, 44, 45, 46, 47, 8, 9, 10, 11,
                                12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 16, 17,
                                18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63,
                                24, 25, 26, 27, 28, 29, 30, 31])

        return remap_channels[chan_num - 1]

    def samples_to_array(self, A, channels=[]):
        """ 
        This will take data matrix A, and convert it into a numpy array, 
        there are three samples of 64 channels in this matrix, however their 
        channels do need to be re-mapped
        """

        if channels == []:
            channels = np.arange(64) + 1
        else:
            channels = np.asarray(channels)

        A = np.asarray(A)

        sample_num = int(len(A) / 64)  # get the sample numbers

        # creating a 64x3 array of zeros (64 channels, 3 samples)
        sample_array = np.zeros((len(channels), sample_num))  

        for i, channel in enumerate(channels):
            sample_array[i, :] = A[self.get_sample_indices(channel, sample_num)]

        return sample_array

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
        """ Read gain for each channel from .set file and return in list of integers """

        gain_list = []

        with open(self.set_file, encoding='cp1252') as f:
            for line in f:
                if line.startswith('gain_ch'):
                    gain_list.append(int(re.findall(r'\d*', line.split(' ')[1])[0]))
                    
        return gain_list

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
        group id
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
        TODO 
        placeholder function, filled with example code
        """
        # creating units channels
        # This is mandatory!!!!
        # Note that if there is no waveform at all in the file
        # then wf_units/wf_gain/wf_offset/wf_left_sweep/wf_sampling_rate
        # can be set to any value because _spike_raw_waveforms
        # will return None
        unit_channels = []
        for c in range(3):
            unit_name = 'unit{}'.format(c)
            unit_id = '#{}'.format(c)
            wf_units = 'uV'
            wf_gain = 1000. / 2 ** 16
            wf_offset = 0.
            wf_left_sweep = 20
            wf_sampling_rate = 10000.
            unit_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                  wf_offset, wf_left_sweep, wf_sampling_rate))
        
        return  np.array(unit_channels, dtype=_spike_channel_dtype)

    def get_event_chan_header(self):
        """
        TODO
        placeholder function, filled with example code
        """
        # creating event/epoch channel
        # This is mandatory!!!!
        # In RawIO epoch and event they are dealt the same way.
        event_channels = []
        event_channels.append(('Some events', 'ev_0', 'event'))
        event_channels.append(('Some epochs', 'ep_1', 'epoch'))

        return np.array(event_channels, dtype=_event_channel_dtype)
