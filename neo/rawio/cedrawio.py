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
        sig_channels = []
        for chan_ind in range(smrx.MaxChannels()):
            print()
            print('chan_ind', chan_ind)
            chan_type = smrx.ChannelType(chan_ind)
            print(chan_type)
            if chan_type == sonpy.lib.DataType.Adc:
                physical_chan = smrx.PhysicalChannel(chan_ind)
                #~ print(chan_type)
                sr = smrx.GetIdealRate(chan_ind)
                #~ print(sr)
                divide = smrx.ChannelDivide(chan_ind)
                #~ print(devide)

                max_time = smrx.ChannelMaxTime(chan_ind)
                print('max_time', max_time)
                first_time = smrx.FirstTime(chan_ind, 0, max_time)
                print('first_time', first_time)
                print('divide', divide)
                print((max_time - first_time)/divide)
                
                # max_times is include so +1
                size_size = (max_time - first_time) /divide +1 
                print('size_size', size_size)
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
                sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, stream_id))
                
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        
        channel_infos = np.array(channel_infos,
                    dtype=[('first_time', 'i8'), ('max_time', 'i8'), ('divide', 'i8'), ('size_size', 'i8')])
        
        # group depend on the signal size
        print(sig_channels)
        print(channel_infos)
        
        
        signal_streams = np.array([('Signals', '0')], dtype=_signal_stream_dtype)


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
        self.header['signal_channels'] = sig_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        return self._t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        return self._num_frames

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._num_frames
            
        #~ self.smrx_file.ReadInts(chan=chan_ind, 
                #~ nMax=int(f.ChannelMaxTime(smrx_ch_ind) / f.ChannelDivide(smrx_ch_ind)),
                #~ tFrom=int(start_frame * f.ChannelDivide(smrx_ch_ind)),
                #~ tUpto=int(end_frame * f.ChannelDivide(smrx_ch_ind))
                #~ )
        
        #~ if channel_indexes is None:
            #~ channel_indexes = slice(self._num_channels)

        #~ raw_signals = self._recgen.recordings[i_start:i_stop, channel_indexes]
        #~ return raw_signals

