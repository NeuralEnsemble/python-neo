"""
Class for reading data from CED (Cambridge Electronic Design)
http://ced.co.uk/

This read *.smrx (and *.smr) from spike2 and signal software.

Note Spike2RawIO/Spike2IO is the old implementation in neo.
It still works without any dependency and should be faster.
Spike2IO only works for smr (32 bit) and not for smrx (64 bit) files.

This implementation depends on the SONPY package:
https://pypi.org/project/sonpy/

Please note that the SONPY package:
  * is NOT open source
  * internally uses a list instead of numpy.ndarray, potentially causing slow data reading
  *  is maintained by CED


Author : Samuel Garcia
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)

import numpy as np
from copy import deepcopy



class CedRawIO(BaseRawIO):
    """
    Class for reading data from CED (Cambridge Electronic Design) spike2.
    This internally uses the sonpy package which is closed source.

    This IO reads smr and smrx files
    """
    extensions = ['smr', 'smrx']
    rawmode = 'one-file'

    def __init__(self, filename='', take_ideal_sampling_rate=False, ):
        BaseRawIO.__init__(self)
        self.filename = filename

        self.take_ideal_sampling_rate = take_ideal_sampling_rate

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        import sonpy

        self.smrx_file = sonpy.lib.SonFile(sName=str(self.filename), bReadOnly=True)
        smrx = self.smrx_file

        self._time_base = smrx.GetTimeBase()

        channel_infos = []
        signal_channels = []
        spike_channels = []
        self._all_spike_ticks = {}

        for chan_ind in range(smrx.MaxChannels()):
            chan_type = smrx.ChannelType(chan_ind)
            chan_id = str(chan_ind)
            if chan_type == sonpy.lib.DataType.Adc:
                physical_chan = smrx.PhysicalChannel(chan_ind)
                divide = smrx.ChannelDivide(chan_ind)
                if self.take_ideal_sampling_rate:
                    sr = smrx.GetIdealRate(chan_ind)
                else:
                    sr = 1. / (smrx.GetTimeBase() * divide)
                max_time = smrx.ChannelMaxTime(chan_ind)
                first_time = smrx.FirstTime(chan_ind, 0, max_time)
                # max_times is included so +1
                time_size = (max_time - first_time) / divide + 1
                channel_infos.append((first_time, max_time, divide, time_size, sr))
                gain = smrx.GetChannelScale(chan_ind) / 6553.6
                offset = smrx.GetChannelOffset(chan_ind)
                units = smrx.GetChannelUnits(chan_ind)
                ch_name = smrx.GetChannelTitle(chan_ind)

                dtype = 'int16'
                # set later after grouping
                stream_id = '0'
                signal_channels.append((ch_name, chan_id, sr, dtype,
                                        units, gain, offset, stream_id))

            elif chan_type == sonpy.lib.DataType.AdcMark:
                # spike and waveforms : only spike times is used here
                ch_name = smrx.GetChannelTitle(chan_ind)
                first_time = smrx.FirstTime(chan_ind, 0, max_time)
                max_time = smrx.ChannelMaxTime(chan_ind)
                divide = smrx.ChannelDivide(chan_ind)
                # here we don't use filter (sonpy.lib.MarkerFilter()) so we get all marker
                wave_marks = smrx.ReadWaveMarks(chan_ind, int(max_time / divide), 0, max_time)

                # here we load in memory all spike once because the access is really slow
                # with the ReadWaveMarks
                spike_ticks = np.array([t.Tick for t in wave_marks])
                spike_codes = np.array([t.Code1 for t in wave_marks])

                unit_ids = np.unique(spike_codes)
                for unit_id in unit_ids:
                    name = f'{ch_name}#{unit_id}'
                    spike_chan_id = f'ch{chan_id}#{unit_id}'
                    spike_channels.append((name, spike_chan_id, '', 1, 0, 0, 0))
                    mask = spike_codes == unit_id
                    self._all_spike_ticks[spike_chan_id] = spike_ticks[mask]

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # channels are grouped into stream if they have a common start, stop, size, divide and sampling_rate
        channel_infos = np.array(channel_infos,
                    dtype=[('first_time', 'i8'), ('max_time', 'i8'),
                           ('divide', 'i8'), ('size', 'i8'), ('sampling_rate', 'f8')])
        unique_info = np.unique(channel_infos)
        self.stream_info = unique_info
        signal_streams = []
        for i, info in enumerate(unique_info):
            stream_id = str(i)
            mask = channel_infos == info
            signal_channels['stream_id'][mask] = stream_id
            num_chans = np.sum(mask)
            stream_name = f'{stream_id} {num_chans}chans'
            signal_streams.append((stream_name, stream_id))
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        # spike channels not handled
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # event channels not handled
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self._seg_t_start = np.inf
        self._seg_t_stop = -np.inf
        for info in self.stream_info:
            self._seg_t_start = min(self._seg_t_start,
                                    info['first_time'] * self._time_base)

            self._seg_t_stop = max(self._seg_t_stop,
                                   info['max_time'] * self._time_base)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return self._seg_t_start

    def _segment_t_stop(self, block_index, seg_index):
        return self._seg_t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        size = self.stream_info[stream_index]['size']
        return size

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        info = self.stream_info[stream_index]
        t_start = info['first_time'] * self._time_base
        return t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.stream_info[stream_index]['size']

        stream_id = self.header['signal_streams']['id'][stream_index]
        signal_channels = self.header['signal_channels']
        mask = signal_channels['stream_id'] == stream_id
        signal_channels = signal_channels[mask]
        if channel_indexes is not None:
            signal_channels = signal_channels[channel_indexes]

        num_chans = len(signal_channels)

        size = i_stop - i_start
        sigs = np.zeros((size, num_chans), dtype='int16')

        info = self.stream_info[stream_index]
        t_from = info['first_time'] + info['divide'] * i_start
        t_upto = info['first_time'] + info['divide'] * i_stop

        for i, chan_id in enumerate(signal_channels['id']):
            chan_ind = int(chan_id)
            sig = self.smrx_file.ReadInts(chan=chan_ind,
                    nMax=size, tFrom=t_from, tUpto=t_upto)
            sigs[:, i] = sig

        return sigs

    def _spike_count(self, block_index, seg_index, unit_index):
        unit_id = self.header['spike_channels'][unit_index]['id']
        spike_ticks = self._all_spike_ticks[unit_id]
        return spike_ticks.size

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        unit_id = self.header['spike_channels'][unit_index]['id']
        spike_ticks = self._all_spike_ticks[unit_id]
        if t_start is not None:
            tick_start = int(t_start / self._time_base)
            spike_ticks = spike_ticks[spike_ticks >= tick_start]
        if t_stop is not None:
            tick_stop = int(t_stop / self._time_base)
            spike_ticks = spike_ticks[spike_ticks <= tick_stop]
        return spike_ticks

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times *= self._time_base
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index,
                                 spike_channel_index, t_start, t_stop):
        return None
