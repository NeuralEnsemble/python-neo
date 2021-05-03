"""
Class for reading data from CED (Cambridge Electronic Design)
http://ced.co.uk/

This read *.smr and *.smrx from spike2 and signal software.

Note Spike2RawIO/Spike2IO is the old implementation in neo.
It still works without any dependency.

This implementation depend on SONPY package
https://pypi.org/project/sonpy/

Please note that the "sonpy" package IS NOT open source.


Author : Samuel Garcia
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)

import numpy as np
from copy import deepcopy

try:
    import sonpy
    HAVE_SONPY = True
except ImportError:
    HAVE_SONPY = False



class CedRawIO(BaseRawIO):
    """
    Class for reading data from CED (Cambridge Electronic Design) spike2.
    
    This read smr and smrx.
    """
    extensions = ['smr', 'smrx']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        assert HAVE_SONPY, 'sonpy must be installed'

        self.smrx_file = sonpy.lib.SonFile(sName=str(self.filename), bReadOnly=True)
        smrx = self.smrx_file
        
        channel_infos = []
        signal_channels = []
        for chan_ind in range(smrx.MaxChannels()):
            #~ print()
            #~ print('chan_ind', chan_ind)
            chan_type = smrx.ChannelType(chan_ind)
            #~ print(chan_type)
            if chan_type == sonpy.lib.DataType.Adc:
                physical_chan = smrx.PhysicalChannel(chan_ind)
                #~ print(chan_type)
                sr = smrx.GetIdealRate(chan_ind)
                #~ print(sr)
                divide = smrx.ChannelDivide(chan_ind)
                #~ print(devide)

                max_time = smrx.ChannelMaxTime(chan_ind)
                #~ print('max_time', max_time)
                first_time = smrx.FirstTime(chan_ind, 0, max_time)
                #~ print('first_time', first_time)
                #~ print('divide', divide)
                #~ print((max_time - first_time)/divide)
                
                # max_times is include so +1
                size_size = (max_time - first_time) /divide +1 
                #~ print('size_size', size_size)
                channel_infos.append((first_time, max_time, divide, size_size))
                
                #~ after_time = smrx.FirstTime(chan_ind, first_time+1, max_time)
                #~ print('after_time', after_time)
                
                
                
                gain = smrx.GetChannelScale(chan_ind)
                #~ print(gain)
                offset = smrx.GetChannelOffset(chan_ind)
                #~ print(offset)
                units = smrx.GetChannelUnits(chan_ind)
                #~ print(units)
                
                ch_name = smrx.GetChannelTitle(chan_ind)
                #~ print(title)
                
                
                chan_id = str(chan_ind)
                dtype = 'int16'
                stream_id = '0'
                signal_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, stream_id))
                
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        
        # cahnnel are group into stream if same start/stop/size/divide
        channel_infos = np.array(channel_infos,
                    dtype=[('first_time', 'i8'), ('max_time', 'i8'), ('divide', 'i8'), ('size', 'i8')])
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
        spike_channels = []
        spike_channels = np.array([], dtype=_spike_channel_dtype)
        
        # event channels not handled
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        
        # TODO
        self._t_stop = 10.
        
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        return self._t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        size = self.stream_info[stream_index]['size']
        return size

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

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
        
        info = self.stream_info[stream_index]
        
        size = i_stop - i_start
        tFrom = info['first_time'] + info['divide'] * i_start
        tUpto = info['first_time'] + info['divide'] * (i_stop )
        sigs = np.zeros((size, num_chans), dtype='int16')
        for i, chan_id in enumerate(signal_channels['id']):
            chan_ind = int(chan_id)
            sig = self.smrx_file.ReadInts(chan=chan_ind, 
                    nMax=size, tFrom=tFrom, tUpto=tUpto )
            sigs[:, i] = sig
        
        return sigs

