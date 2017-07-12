# -*- coding: utf-8 -*-
"""
Class for reading data from BrainVision product.

This code was originally made by L. Pezard (2010), modified B. Burle and
S. More.

Supported : Read

Author: sgarcia
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype, 
        _event_channel_dtype)

import numpy as np

import datetime
import os
import re



class BrainVisionRawIO(BaseRawIO):
    """
    
    """
    extensions = ['vhdr']
    
    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename 
    
    def _parse_header(self):
       # # Read header file (vhdr)
        header = read_brain_soup(self.filename)

        assert header['Common Infos'][
            'DataFormat'] == 'BINARY', NotImplementedError
        assert header['Common Infos'][
            'DataOrientation'] == 'MULTIPLEXED', NotImplementedError

        nb_channel = int(header['Common Infos']['NumberOfChannels'])
        sr=1.e6 /float(header['Common Infos']['SamplingInterval'])
        self._sampling_rate = sr

        fmt = header['Binary Infos']['BinaryFormat']
        fmts = { 'INT_16':np.int16,  'INT_32':np.int32, 'IEEE_FLOAT_32':np.float32,}

        assert fmt in fmts, NotImplementedError
        dt = fmts[fmt]


        #raw signals memmap
        binary_file = os.path.splitext(self.filename)[0] + '.eeg'
        sigs = np.memmap(binary_file, dtype=dt, mode='r', offset=0 )
        if sigs.size%nb_channel!=0:
            sigs = sigs[:-sigs.size%nb_channel]
        self._raw_signals = sigs.reshape(-1, nb_channel)
        
        sig_channels = []
        for c in range(nb_channel):
            name, ref, res, units = header['Channel Infos'][
                'Ch%d' % (c + 1,)].split(',')
            units =units.replace('µ', 'u')
            chan_id = c+1
            if dt == np.int16 or dt == np.int32:
                gain = float(res)
            else:
                gain = 1
            offset = 0
            sig_channels.append((name, chan_id, units, gain,offset))
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        
        
        #No spikes
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        
        #TODO marker
        # read marker
        #~ marker_file = os.path.splitext(self.filename)[0] + '.vmrk'
        #~ all_info = read_brain_soup(marker_file)['Marker Infos']
        #~ all_types = []
        #~ times = []
        #~ labels = []
        #~ for i in range(len(all_info)):
            #~ type_, label, pos, size, channel = all_info[
                #~ 'Mk%d' % (i + 1,)].split(',')[:5]
            #~ all_types.append(type_)
            #~ times.append(float(pos) / self._sampling_rate)
            #~ labels.append(label)
        #~ all_types = np.array(all_types)
        #~ times = np.array(times)
        #~ labels = np.array(labels, dtype='S')
        #~ for type_ in np.unique(all_types):
            #~ ind = type_ == all_types
            #~ if lazy:
                #~ ea = Event(name=str(type_))
                #~ ea.lazy_shape = -1
            #~ else:
                #~ ea = Event(
                    #~ times=times[ind], labels=labels[ind], name=str(type_))
            #~ seg.events.append(ea)
        
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        
        #fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels
        
        self._generate_empty_annotations()
    
    def _source_name(self):
        return self.filename
    
    def _block_count(self):
        return 1
    
    def _segment_count(self, block_index):
        return 1
    
    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._raw_signals.shape[0]/self._sampling_rate
        return t_stop
    
    ###
    def _analogsignal_shape(self, block_index, seg_index):
        return self._raw_signals.shape
    
    def _analogsignal_sampling_rate(self):
        return self._sampling_rate

    def _get_analogsignal_chunk(self, block_index, seg_index,  i_start, i_stop, channel_indexes):
        raw_signals = self._raw_signals[slice(i_start, i_stop), channel_indexes]
        return raw_signals
    
    ###
    def _spike_count(self,  block_index, seg_index, unit_index):
        return 0
   
    ###
    # event and epoch zone
    def _event_count(self, block_index, seg_index, event_channel_index):
        raise(NotImplementedError)
    
    def _event_timestamps(self,  block_index, seg_index, event_channel_index, t_start, t_stop):
        raise(NotImplementedError)
    
    def _rescale_event_timestamp(self, event_timestamps, dtype):
        raise(NotImplementedError)
    
    def _rescale_epoch_duration(self, raw_duration, dtype):
        raise(NotImplementedError)




def read_brain_soup(filename):
    with open(filename, 'r') as f:
        section = None
        all_info = {}
        for line in f:
            line = line.strip('\n').strip('\r')
            if line.startswith('['):
                section = re.findall('\[([\S ]+)\]', line)[0]
                all_info[section] = {}
                continue
            if line.startswith(';'):
                continue
            if '=' in line and len(line.split('=')) == 2:
                k, v = line.split('=')
                all_info[section][k] = v
    
    return all_info
