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
from typing import Tuple, TypedDict, Union

import numpy as np

from .baserawio import (BaseRawIO, _event_channel_dtype, _signal_channel_dtype,
                        _signal_stream_dtype, _spike_channel_dtype)


class ExampleRawIOHeader(TypedDict):
    nb_block: int
    nb_segment: list[int]
    signal_streams: np.ndarray  # with dtype `_signal_stream_dtype`
    signal_channels: np.ndarray  # with dtype `_signal_channel_dtype`
    spike_channels: np.ndarray  # with dtype `_spike_channel_dtype`
    event_channels: np.ndarray  # width dtype `_event_channel_dtype`


class ExampleRawIO(BaseRawIO):
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
    extensions = ['fake']
    rawmode = 'one-file'

    def __init__(self, filename: Union[str, pathlib.Path] = ''):
        BaseRawIO.__init__(self)
        # note that this filename is ued in self._source_name
        self.filename = filename

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

        # create fake signals stream information
        signal_streams = []
        for c in range(2):
            name = f'stream {c}'
            stream_id = c
            signal_streams.append((name, stream_id))
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        # create fake signals channels information
        # This is mandatory!!!!
        # gain/offset/units are really important because
        # the scaling to real value will be done with that
        # The real signal will be evaluated as `(raw_signal * gain + offset) * pq.Quantity(units)`
        signal_channels = []
        for c in range(16):
            ch_name = 'ch{}'.format(c)
            # our channel id is c+1 just for fun
            # Note that chan_id should be related to
            # original channel id in the file format
            # so that the end user should not be lost when reading datasets
            chan_id = c + 1
            sr = 10000.  # Hz
            dtype = 'int16'
            units = 'uV'
            gain = 1000. / 2 ** 16
            offset = 0.
            # stream_id indicates how to group channels
            # channels inside a "stream" share same characteristics
            #  (sampling rate/dtype/t_start/units/...)
            stream_id = str(c // 8)
            signal_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, stream_id))
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # A stream can contain signals with different physical units.
        # Here, the two last channels will have different units (pA)
        # Since AnalogSignals must have consistent units across channels,
        # this stream will be split in 2 parts on the neo.io level and finally 3 AnalogSignals
        # will be generated per Segment.
        signal_channels[-2:]['units'] = 'pA'

        # create fake units channels
        # This is mandatory!!!!
        # Note that if there is no waveform at all in the file
        # then wf_units/wf_gain/wf_offset/wf_left_sweep/wf_sampling_rate
        # can be set to any value because _spike_raw_waveforms
        # will return None
        spike_channels = []
        for c in range(3):
            unit_name = 'unit{}'.format(c)
            unit_id = '#{}'.format(c)
            wf_units = 'uV'
            wf_gain = 1000. / 2 ** 16
            wf_offset = 0.
            wf_left_sweep = 20
            wf_sampling_rate = 10000.
            spike_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                  wf_offset, wf_left_sweep, wf_sampling_rate))
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # creating event/epoch channel
        # This is mandatory!!!!
        # In RawIO epoch and event they are dealt the same way.
        event_channels = []
        event_channels.append(('Some events', 'ev_0', 'event'))
        event_channels.append(('Some epochs', 'ep_1', 'epoch'))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # fille into header dict
        # This is mandatory!!!!!
        self.header = ExampleRawIOHeader()
        self.header['nb_block'] = 2
        self.header['nb_segment'] = [2, 3]
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

        # Until here all mandatory operations for setting up a rawio are implemented.
        # The following lines provide additional, recommended annotations for the
        # final neo objects.
        for block_index in range(2):
            bl_ann = self.raw_annotations['blocks'][block_index]
            bl_ann['name'] = 'Block #{}'.format(block_index)
            bl_ann['block_extra_info'] = 'This is the block {}'.format(block_index)
            for seg_index in range([2, 3][block_index]):
                seg_ann = bl_ann['segments'][seg_index]
                seg_ann['name'] = 'Seg #{} Block #{}'.format(
                    seg_index, block_index)
                seg_ann['seg_extra_info'] = 'This is the seg {} of block {}'.format(
                    seg_index, block_index)
                for c in range(2):
                    sig_an = seg_ann['signals'][c]['nickname'] = \
                        f'This stream {c} is from a subdevice'
                    # add some array annotations (8 channels)
                    sig_an = seg_ann['signals'][c]['__array_annotations__']['impedance'] = \
                        np.random.rand(8) * 10000
                for c in range(3):
                    spiketrain_an = seg_ann['spikes'][c]
                    spiketrain_an['quality'] = 'Good!!'
                    # add some array annotations
                    num_spikes = self.spike_count(block_index, seg_index, c)
                    spiketrain_an['__array_annotations__']['amplitudes'] = \
                        np.random.randn(num_spikes)

                for c in range(2):
                    event_an = seg_ann['events'][c]
                    if c == 0:
                        event_an['nickname'] = 'Miss Event 0'
                        # add some array annotations
                        num_ev = self.event_count(block_index, seg_index, c)
                        event_an['__array_annotations__']['button'] = ['A'] * num_ev
                    elif c == 1:
                        event_an['nickname'] = 'MrEpoch 1'

    def _segment_t_start(self, block_index: int, seg_index: int) -> float:
        # this must return an float scale in second
        # this t_start will be shared by all object in the segment
        # except AnalogSignal
        all_starts = [[0., 15.], [0., 20., 60.]]
        return all_starts[block_index][seg_index]

    def _segment_t_stop(self, block_index: int, seg_index: int) -> float:
        # this must return an float scale in second
        all_stops = [[10., 25.], [10., 30., 70.]]
        return all_stops[block_index][seg_index]

    def _get_signal_size(self, block_index: int, seg_index: int, stream_index: int) -> int:
        # We generate fake data in which the two stream signals have the same shape
        # across all segments (10.0 seconds)
        # This is not the case for real data, instead you should return the signal
        # size depending on the block_index and segment_index
        # this must return an int = the number of sample

        # Note that channel_indexes can be ignored for most cases
        # except for several sampling rate.
        return 100000

    def _get_signal_t_start(self, block_index: int, seg_index: int, stream_index: int) -> float:
        # This give the t_start of signals.
        # Very often this equal to _segment_t_start but not
        # always.
        # this must return an float scale in second

        # Note that channel_indexes can be ignored for most cases
        # except for several sampling rate.

        # Here this is the same.
        # this is not always the case
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index: int, seg_index: int,
                                i_start: Union[int, None], i_stop: Union[int, None],
                                stream_index: int,
                                channel_indexes: Union[np.ndarray,
                                                       list, slice, None]
                                ) -> np.ndarray:
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

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = 100000

        if i_start < 0 or i_stop > 100000:
            # some check
            raise IndexError("I don't like your jokes")

        if channel_indexes is None:
            nb_chan = 8
        elif isinstance(channel_indexes, slice):
            channel_indexes = np.arange(8, dtype='int')[channel_indexes]
            nb_chan = len(channel_indexes)
        else:
            channel_indexes = np.asarray(channel_indexes)
            if any(channel_indexes < 0):
                raise IndexError('bad boy')
            if any(channel_indexes >= 8):
                raise IndexError('big bad wolf')
            nb_chan = len(channel_indexes)

        raw_signals = np.zeros((i_stop - i_start, nb_chan), dtype='int16')
        return raw_signals

    def _spike_count(self, block_index: int, seg_index: int, spike_channel_index: int) -> int:
        # Must return the nb of spikes for given (block_index, seg_index, spike_channel_index)
        # we are lucky:  our units have all the same nb of spikes!!
        # it is not always the case
        nb_spikes = 20
        return nb_spikes

    def _get_spike_timestamps(self, block_index: int, seg_index: int, spike_channel_index: int,
                              t_start: Union[float, None], t_stop: Union[float, None]) -> np.ndarray:
        # In our IO, timestamp are internally coded 'int64' and they
        # represent the index of the signals 10kHz
        # we are lucky: spikes have the same discharge in all segments!!
        # incredible neuron!! This is not always the case

        # the same clip t_start/t_start must be used in _spike_raw_waveforms()

        ts_start = (self._segment_t_start(block_index, seg_index) * 10000)

        spike_timestamps = np.arange(0, 10000, 500) + ts_start

        if t_start is not None or t_stop is not None:
            # restrict spikes to given limits (in seconds)
            lim0 = int(t_start * 10000)
            lim1 = int(t_stop * 10000)
            mask = (spike_timestamps >= lim0) & (spike_timestamps <= lim1)
            spike_timestamps = spike_timestamps[mask]

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps: np.dtype, dtype: np.dtype) -> np.ndarray:
        # must rescale to second a particular spike_timestamps
        # with a fixed dtype so the user can choose the precision he want.
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= 10000.  # because 10kHz
        return spike_times

    def _get_spike_raw_waveforms(self, block_index: int, seg_index: int, spike_channel_index: int,
                                 t_start: Union[float, None], t_stop: Union[float, None]) -> np.ndarray:
        # this must return a 3D numpy array (nb_spike, nb_channel, nb_sample)
        # in the original dtype
        # this must be as fast as possible.
        # the same clip t_start/t_start must be used in _spike_timestamps()

        # If there there is no waveform supported in the
        # IO them _spike_raw_waveforms must return None

        # In our IO waveforms come from all channels
        # they are int16
        # conversion to real units is done with self.header['spike_channels']
        # Here, we have a realistic case: all waveforms are only noise.
        # it is not always the case
        # we 20 spikes with a sweep of 50 (5ms)

        # trick to get how many spike in the slice
        ts = self._get_spike_timestamps(block_index, seg_index,
                                        spike_channel_index, t_start, t_stop)
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

    def _get_event_timestamps(self, block_index: int, seg_index: int, event_channel_index: int,
                              t_start: Union[float, None], t_stop: Union[float, None]
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _rescale_event_timestamp(self, event_timestamps: np.ndarray, dtype: np.dtype,
                                 event_channel_index: int) -> np.ndarray:
        # must rescale to second a particular event_timestamps
        # with a fixed dtype so the user can choose the precision he want.

        # really easy here because in our case it is already seconds
        event_times = event_timestamps.astype(dtype)
        return event_times

    def _rescale_epoch_duration(self, raw_duration: np.ndarray, dtype: np.dtype,
                                event_channel_index: int):
        # really easy here because in our case it is already seconds
        durations = raw_duration.astype(dtype)
        return durations
