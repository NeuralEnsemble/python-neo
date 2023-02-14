"""
ExampleRawIO is a class of a  fake example.
This is to be used when coding a new RawIO.


Rules for creating a new class:
  1. Step 1: Create the main class
    * Create a file in **neo/rawio/** that endith with "rawio.py"
    * Create the class that inherits from BaseRawIO
    * copy/paste all methods that need to be implemented.
    * code hard! The main difficulty is `_parse_header()`.
      In short you have to create a mandatory dict that
      contains channel informations::

            self.header = {}
            self.header['nb_block'] = 2
            self.header['nb_segment'] = [2, 3]
            self.header['signal_streams'] = signal_streams
            self.header['signal_channels'] = signal_channels
            self.header['spike_channels'] = spike_channels
            self.header['event_channels'] = event_channels

  2. Step 2: RawIO test:
    * create a file in neo/rawio/tests with the same name with "test_" prefix
    * copy paste neo/rawio/tests/test_examplerawio.py and do the same

  3. Step 3 : Create the neo.io class with the wrapper
    * Create a file in neo/io/ that ends with "io.py"
    * Create a class that inherits both your RawIO class and BaseFromRaw class
    * copy/paste from neo/io/exampleio.py

  4.Step 4 : IO test
    * create a file in neo/test/iotest with the same previous name with "test_" prefix
    * copy/paste from neo/test/iotest/test_exampleio.py



"""
import pathlib
import warnings
import numpy as np
from collections import namedtuple
from ..baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                         _spike_channel_dtype, _event_channel_dtype)


class Plexon2RawIO(BaseRawIO):
    """
    Class for "reading" fake data from an imaginary file.

    For the user, it gives access to raw data (signals, event, spikes) as they
    are in the (fake) file int16 and int64.

    For a developer, it is just an example showing guidelines for someone who wants
    to develop a new IO module.

    Two rules for developers:
      * Respect the :ref:`neo_rawio_API`
      * Follow the :ref:`io_guiline`

    This fake IO:
        * has 2 blocks
        * blocks have 2 and 3 segments
        * has  2 signals streams  of 8 channel each (sample_rate = 10000) so 16 channels in total
        * has 3 spike_channels
        * has 2 event channels: one has *type=event*, the other has
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
        >>> spike_timestamp = reader.spike_timestamps(spike_channel_index=0,
                            t_start=None, t_stop=None)
        >>> spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
        >>> ev_timestamps, _, ev_labels = reader.event_timestamps(event_channel_index=0)

    """
    extensions = ['pl2']
    rawmode = 'one-file'

    def __init__(self, filename='', pl2_dll_path=None):

        # note that this filename is used in self._source_name
        self.filename = pathlib.Path(filename)
        self._analogsignal_cache = {}

        if (not self.filename.exists()) or (not self.filename.is_file()):
            raise ValueError(f'{self.filename} is not a file.')

        from neo.rawio.plexon2rawio.pypl2.pypl2lib import PyPL2FileReader, PL2FileInfo
        BaseRawIO.__init__(self)

        self.pl2reader = PyPL2FileReader(pl2_dll_path=pl2_dll_path)

        # Open the file.
        self.pl2reader.pl2_open_file(self.filename)

    def _source_name(self):
        # this function is used by __repr__
        # for general cases self.filename is good
        # But for URL you could mask some part of the URL to keep
        # the main part.
        return self.filename

    def _parse_header(self):
        # This is the central part of a RawIO
        # we need to collect from the original format all
        # information required for fast access
        # at any place in the file
        # In short `_parse_header()` can be slow but
        # `_get_analogsignal_chunk()` need to be as fast as possible

        signal_streams = []
        Stream = namedtuple('Stream', 'id name sampling_rate n_samples')

        # The real signal will be evaluated as `(raw_signal * gain + offset) * pq.Quantity(units)`
        signal_channels = []
        stream_characteristics = {}
        for c in range(self.pl2reader.pl2_file_info.m_TotalNumberOfAnalogChannels):
            achannel_info = self.pl2reader.pl2_get_analog_channel_info(c)

            # only consider active channels
            if not achannel_info.m_ChannelEnabled:
                continue

            # assign to matching stream or create new stream based on signal characteristics
            sampling_rate = achannel_info.m_SamplesPerSecond
            n_samples = achannel_info.m_NumberOfValues
            stream_characteristic = (sampling_rate, n_samples)
            if stream_characteristic in stream_characteristics:
                stream_id = stream_characteristics[stream_characteristic].id
            else:
                stream_id = str(len(stream_characteristics))
                stream_name = f'stream@{sampling_rate}Hz'
                new_stream = Stream(stream_id, stream_name, sampling_rate, n_samples)
                stream_characteristics[stream_characteristic] = new_stream

            ch_name = achannel_info.m_Name.decode()
            chan_id = f'source_{achannel_info.m_Source}#channel_{achannel_info.m_Channel}'
            sr = achannel_info.m_SamplesPerSecond  # Hz
            dtype = 'int16' # TODO: Check if this is the correct correspondance to c_short
            units = achannel_info.m_Units.decode()
            gain = achannel_info.m_CoeffToConvertToUnits
            offset = 0.
            stream_id = stream_id
            signal_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, stream_id))

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        self.signal_stream_characteristics = stream_characteristics
        signal_streams = np.array([(name, id) for (id, name, _, _) in stream_characteristics.values()],
                                  dtype=_signal_stream_dtype)

        # create fake units channels
        # This is mandatory!!!!
        # Note that if there is no waveform at all in the file
        # then wf_units/wf_gain/wf_offset/wf_left_sweep/wf_sampling_rate
        # can be set to any value because _spike_raw_waveforms
        # will return None
        # pre-loading spike channel_data for later usage
        self._spikechannel_cache = {}

        spike_channels = []
        for c in range(self.pl2reader.pl2_file_info.m_TotalNumberOfSpikeChannels):
            schannel_info = self.pl2reader.pl2_get_spike_channel_info(c)

            # only consider active channels
            if not schannel_info.m_ChannelEnabled:
                continue

            for channel_unit_id in range(schannel_info.m_NumberOfUnits):
                unit_name = f'{schannel_info.m_Name.decode()}-{channel_unit_id}'
                unit_id = f'#{schannel_info.m_Channel}-{channel_unit_id}'
                wf_units = schannel_info.m_Units
                wf_gain = schannel_info.m_CoeffToConvertToUnits
                wf_offset = 0.
                wf_left_sweep = schannel_info.m_PreThresholdSamples
                wf_sampling_rate = schannel_info.m_SamplesPerSecond
                spike_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                      wf_offset, wf_left_sweep, wf_sampling_rate))

            # pre-loading spiking data
            schannel_name = schannel_info.m_Name.decode()
            self._spikechannel_cache[schannel_name] = self.pl2reader.pl2_get_spike_channel_data_by_name(schannel_name)

        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)



        # creating event/epoch channel
        self._event_channel_cache = {}
        event_channels = []
        for i in range(self.pl2reader.pl2_file_info.m_NumberOfDigitalChannels):
            echannel_info = self.pl2reader.pl2_get_digital_channel_info(i)

            # only consider active channels
            if not echannel_info.m_ChannelEnabled:
                continue

            channel_name = echannel_info.m_Name.decode()
            # event channels are characterized by (name, id, type), with type in ['event', 'epoch']
            event_channels.append((channel_name, echannel_info.m_Channel, 'event'))

            self._event_channel_cache[channel_name] = self.pl2reader.pl2_get_digital_channel_data_by_name(channel_name)

        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # fill into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]  # TODO: Check if pl2 format can contain multiple segments
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        # insert some annotations/array_annotations at some place
        # at neo.io level. IOs can add annotations
        # to any object. To keep this functionality with the wrapper
        # BaseFromRaw you can add annotations in a nested dict.

        # `_generate_minimal_annotations()` must be called to generate the nested
        # dict of annotations/array_annotations
        self._generate_minimal_annotations()
        # this pprint lines really help for understand the nested (and complicated sometimes) dict
        # from pprint import pprint
        # pprint(self.raw_annotations)


        # TODO: pl2_file_info.m_ReprocessorDateTime seems to be empty. Check with original implementation

        # Until here all mandatory operations for setting up a rawio are implemented.
        # The following lines provide additional, recommended annotations for the
        # final neo objects.
        block_index = 0
        bl_ann = self.raw_annotations['blocks'][block_index]
        bl_ann['name'] = 'Block containing PL2 data#{}'.format(block_index)
        bl_ann['file_origin'] = self.filename
        pl2_file_info = {attr: getattr(self.pl2reader.pl2_file_info, attr) for attr, _ in self.pl2reader.pl2_file_info._fields_}
        bl_ann.update(pl2_file_info)
        for seg_index in range(1): # TODO: Check if PL2 file can contain multiple segments
            seg_ann = bl_ann['segments'][seg_index]
            # seg_ann['name'] = 'Seg #{} Block #{}'.format(
            #     seg_index, block_index)
            # seg_ann['seg_extra_info'] = 'This is the seg {} of block {}'.format(
            #     seg_index, block_index)
            # for c in range(2):
            #     sig_an = seg_ann['signals'][c]['nickname'] = \
            #         f'This stream {c} is from a subdevice'
            #     # add some array annotations (8 channels)
            #     sig_an = seg_ann['signals'][c]['__array_annotations__']['impedance'] = \
            #         np.random.rand(8) * 10000
            # for c in range(3):
            #     spiketrain_an = seg_ann['spikes'][c]
            #     spiketrain_an['quality'] = 'Good!!'
            #     # add some array annotations
            #     num_spikes = self.spike_count(block_index, seg_index, c)
            #     spiketrain_an['__array_annotations__']['amplitudes'] = \
            #         np.random.randn(num_spikes)

            # for c in range(2):
            #     event_an = seg_ann['events'][c]
            #     if c == 0:
            #         event_an['nickname'] = 'Miss Event 0'
            #         # add some array annotations
            #         num_ev = self.event_count(block_index, seg_index, c)
            #         event_an['__array_annotations__']['button'] = ['A'] * num_ev
            #     elif c == 1:
            #         event_an['nickname'] = 'MrEpoch 1'

    def _segment_t_start(self, block_index, seg_index):
        # this must return an float scale in second
        # this t_start will be shared by all object in the segment
        # except AnalogSignal
        return self.pl2reader.pl2_file_info.m_StartRecordingTime / self.pl2reader.pl2_file_info.m_TimestampFrequency

    def _segment_t_stop(self, block_index, seg_index):
        # this must return an float scale in second
        end_time = self.pl2reader.pl2_file_info.m_StartRecordingTime + self.pl2reader.pl2_file_info.m_DurationOfRecording
        return end_time / self.pl2reader.pl2_file_info.m_TimestampFrequency


    def _get_signal_size(self, block_index, seg_index, stream_index):
        # this must return an int = the number of sample

        stream_id = self.header['signal_streams'][stream_index]['id']
        stream_characteristic = list(self.signal_stream_characteristics.values())[stream_index]
        assert stream_id == stream_characteristic.id
        return stream_characteristic.n_samples

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        # This give the t_start of signals.
        # this must return an float scale in second

        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        # this must return a signal chunk in a signal stream
        # limited with i_start/i_stop (can be None)
        # channel_indexes can be None (=all channel in the stream) or a list or numpy.array
        # This must return a numpy array 2D (even with one channel).
        # This must return the original dtype. No conversion here.
        # This must as fast as possible.
        # To speed up this call all preparatory calculations should be implemented
        # in _parse_header().

        # Here we are lucky:  our signals is always zeros!!
        # it is not always the case :)
        # internally signals are int16
        # conversion to real units is done with self.header['signal_channels']

        stream_id = self.header['signal_streams'][stream_index]['id']
        mask = self.header['signal_channels']['stream_id'] == stream_id
        stream_channels = self.header['signal_channels'][mask]

        n_channels = len(stream_channels)
        n_samples = self.get_signal_size(block_index, seg_index, stream_index)

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = n_samples

        if i_start < 0 or i_stop > n_samples:
            raise IndexError(f"Indexes ({i_start}, {i_stop}) out of range for signal with {n_samples} samples")

        # converting channel_indexes to array representation
        if channel_indexes is None:
            channel_indexes = np.arange(len(stream_channels), dtype='int')
        elif isinstance(channel_indexes, slice):
            channel_indexes = np.arange(len(stream_channels), dtype='int')[channel_indexes]
        else:
            channel_indexes = np.asarray(channel_indexes)

        if any(channel_indexes < 0) or any(channel_indexes >= n_channels):
            raise IndexError(f'Channel index out of range {channel_indexes} for stream with {n_channels} channels')

        nb_chan = len(channel_indexes)

        raw_signals = np.empty((i_stop - i_start, nb_chan), dtype='int16')
        for i, channel_idx in enumerate(channel_indexes):
            channel_name = stream_channels['name'][channel_idx]

            # use previously loaded channel data if possible
            if channel_name in self._analogsignal_cache:
                values = self._analogsignal_cache[channel_name]
            else:
                fragment_timestamps, fragment_counts, values = self.pl2reader.pl2_get_analog_channel_data_by_name(channel_name)
                self._analogsignal_cache[channel_name] = values

            raw_signals[:, i] = values[i_start: i_stop]

        return raw_signals

    def clean_analogsignal_cache(self):
        for channel_name, values in self._analogsignal_cache.items():
            del values
        self._analogsignal_cache = {}

    def _spike_count(self, block_index, seg_index, spike_channel_index):
        channel_header = self.header['spike_channels'][spike_channel_index]
        channel_name, channel_unit_id = channel_header['name'].split('-')
        channel_unit_id = int(channel_unit_id)

        spike_timestamps, unit_ids, waveforms = self._spikechannel_cache[channel_name]
        nb_spikes = np.count_nonzero(unit_ids == channel_unit_id)

        return nb_spikes

    def _get_spike_timestamps(self, block_index, seg_index, spike_channel_index, t_start, t_stop):
        channel_header = self.header['spike_channels'][spike_channel_index]
        channel_name, channel_unit_id = channel_header['name'].split('-')
        channel_unit_id = int(channel_unit_id)

        spike_timestamps, unit_ids, waveforms = self._spikechannel_cache[channel_name]

        if t_start is not None or t_stop is not None:
            # restrict spikes to given limits (in seconds)
            timestamp_frequency = self.pl2reader.pl2_file_info.m_TimestampFrequency
            lim0 = int(t_start * timestamp_frequency)
            lim1 = int(t_stop * self.pl2reader.pl2_file_info.m_TimestampFrequency)
            time_mask = (spike_timestamps >= lim0) & (spike_timestamps <= lim1)
        else:
            time_mask = slice(None, None)

        unit_mask = unit_ids[time_mask] == channel_unit_id
        spike_timestamps = spike_timestamps[time_mask][unit_mask]

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        # must rescale to second a particular spike_timestamps
        # with a fixed dtype so the user can choose the precision they want
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self.pl2reader.pl2_file_info.m_TimestampFrequency
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, spike_channel_index,
                                 t_start, t_stop):
        # this must return a 3D numpy array (nb_spike, nb_channel, nb_sample)
        # in the original dtype
        # this must be as fast as possible.
        # the same clip t_start/t_start must be used in _spike_timestamps()

        # If there there is no waveform supported in the
        # IO them _spike_raw_waveforms must return None

        channel_header = self.header['spike_channels'][spike_channel_index]
        channel_name, channel_unit_id = channel_header['name'].split('-')

        spike_timestamps, unit_ids, waveforms = self._spikechannel_cache[channel_name]

        if t_start is not None or t_stop is not None:
            # restrict spikes to given limits (in seconds)
            timestamp_frequency = self.pl2reader.pl2_file_info.m_TimestampFrequency
            lim0 = int(t_start * timestamp_frequency)
            lim1 = int(t_stop * self.pl2reader.pl2_file_info.m_TimestampFrequency)
            time_mask = (spike_timestamps >= lim0) & (spike_timestamps <= lim1)
        else:
            time_mask = slice(None, None)

        unit_mask = unit_ids[time_mask] == int(channel_unit_id)
        waveforms = waveforms[time_mask][unit_mask]

        # add tetrode dimension
        waveforms = np.expand_dims(waveforms, axis=1)
        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):

        channel_header = self.header['event_channels'][event_channel_index]
        channel_name = channel_header['name']

        event_times, values = self._event_channel_cache[channel_name]

        return len(event_times)

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        # the main difference between spike channel and event channel
        # is that for here we have 3 numpy array timestamp, durations, labels
        # durations must be None for 'event'
        # label must a dtype ='U'

        channel_header = self.header['event_channels'][event_channel_index]
        channel_name = channel_header['name']

        event_times, labels = self._event_channel_cache[channel_name]
        labels = np.asarray(labels, dtype='U')
        durations = None

        if t_start is not None or t_stop is not None:
            # restrict events to given limits (in seconds)
            timestamp_frequency = self.pl2reader.pl2_file_info.m_TimestampFrequency
            lim0 = int(t_start * timestamp_frequency)
            lim1 = int(t_stop * self.pl2reader.pl2_file_info.m_TimestampFrequency)
            time_mask = (event_times >= lim0) & (event_times <= lim1)
        else:
            time_mask = np.ones_like(event_times)

        durations = None

        return event_times[time_mask], durations, labels[time_mask]

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        # must rescale to second a particular event_timestamps
        # with a fixed dtype so the user can choose the precision he want.

        # really easy here because in our case it is already seconds
        event_times = event_timestamps.astype(dtype)
        event_times /= self.pl2reader.pl2_file_info.m_TimestampFrequency
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        # really easy here because in our case it is already seconds
        durations = raw_duration.astype(dtype)
        durations /= self.pl2reader.pl2_file_info.m_TimestampFrequency
        return durations

    def close(self):
        self.pl2reader.pl2_close_all_files()
