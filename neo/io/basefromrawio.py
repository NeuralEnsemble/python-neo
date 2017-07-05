# -*- coding: utf-8 -*-
"""
BaseFromRaw
======

BaseFromRaw implement a bridge between the new neo.rawio API
and the neo.io legacy that give neo.core object.
The neo.rawio API is more restricted and limited and do not cover tricky
cases with asymetrical tree of neo object.
But if a fromat is done in neo.rawio the neo.io is done for free with this class.


"""

import collections
import logging
import numpy as np

from neo import logging_handler
from neo.core import (AnalogSignal, Block,
                      Epoch, Event,
                      IrregularlySampledSignal,
                      ChannelIndex,
                      Segment, SpikeTrain, Unit)
from neo.io.baseio import BaseIO

import quantities as pq

class BaseFromRaw(BaseIO):
    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal, SpikeTrain, Unit, ChannelIndex, Event, Epoch]
    readable_objects = [Block, Segment]
    writeable_objects = []

    is_streameable = True

    name = 'BaseIO'
    description = ''
    extentions = []

    mode = 'file'
    
    __prefered_signal_group_mode = 'split-all' #'group-by-same-units'
    

    def __init__(self, *args, **kargs):
        BaseIO.__init__(self, *args, **kargs)
        self.parse_header()
    
    #~ def read_all_blocks(self, **kargs):
        #~ blocks = []
        #~ for bl_index in range(self.block_count()):
            #~ bl = self.read_block(block_index=bl_index, **kargs)
            #~ blocks.append(bl)
        #~ return blocks
    
    def read_block(self, block_index=0, lazy=False, cascade=True, signal_group_mode=None,  **kargs):
        
        if signal_group_mode is None:
            signal_group_mode = self.__prefered_signal_group_mode
        
        bl = Block(name='Block {}'.format(block_index))
        if not cascade:
            return bl
        
        channels = self.header['signal_channels']
        for i, ind in self._make_signal_channel_groups(signal_group_mode=signal_group_mode).items():
            channel_index = ChannelIndex(index=ind, channel_names=channels[ind]['name'].astype('S'),
                            channel_ids=channels[ind]['id'], name='Channel group {}'.format(i))
            bl.channel_indexes.append(channel_index)
        
        for seg_index in range(self.segment_count(block_index)):
            seg =  self.read_segment(block_index=block_index, seg_index=seg_index, 
                                                                lazy=lazy, cascade=cascade, signal_group_mode=signal_group_mode, **kargs)
            bl.segments.append(seg)
            
            for i, anasig in enumerate(seg.analogsignals):
                bl.channel_indexes[i].analogsignals.append(anasig)
        
        bl.create_many_to_one_relationship()
        
        return bl

    def read_segment(self, block_index=0, seg_index=0, lazy=False, cascade=True, 
                        signal_group_mode='group-by-same-units', **kargs):
        seg = Segment(index=seg_index)#name, 

        if not cascade:
            return seg
        
        
        #AnalogSignal
        signal_channels = self.header['signal_channels']
        channel_indexes=np.arange(signal_channels.size)
        
        if not lazy:
            raw_signal = self.get_analogsignal_chunk(block_index=block_index, seg_index=seg_index,
                        i_start=None, i_stop=None, channel_indexes=channel_indexes)
            float_signal = self.rescale_signal_raw_to_float(raw_signal,  dtype='float32', channel_indexes=channel_indexes)
        else:
            sig_shape = self.analogsignal_shape(block_index=block_index, seg_index=seg_index,)
        
        sr = self.analogsignal_sampling_rate() * pq.Hz
        t_start = self.segment_t_start(block_index, seg_index) * pq.s
        t_stop = self.segment_t_stop(block_index, seg_index) * pq.s
        for i, ind in self._make_signal_channel_groups(signal_group_mode=signal_group_mode).items():
            units = np.unique(signal_channels[ind]['units'])
            assert len(units)==1
            units = units[0]

            if lazy:
                anasig = AnalogSignal(np.array([]), units=units,  copy=False,
                        sampling_rate=sr, t_start=t_start)
                anasig.lazy_shape = (sig_shape[0], len(ind))
            else:
                anasig = AnalogSignal(float_signal[:, ind], units=units,  copy=False,
                        sampling_rate=sr, t_start=t_start)
            
            seg.analogsignals.append(anasig)
        
        
        #SpikeTrain
        unit_channels = self.header['unit_channels']
        for unit_index in range(len(unit_channels)):
            if not lazy:
                spike_timestamp = self.spike_timestamps(block_index=block_index, seg_index=seg_index, 
                                        unit_index=unit_index, t_start=None, t_stop=None)
                spike_times = self.rescale_spike_timestamp(spike_timestamp, 'float64')
                
                sptr = SpikeTrain(spike_times, units='s', copy=False, t_start=t_start, t_stop=t_stop)
            else:
                nb = self.spike_count(block_index=block_index, seg_index=seg_index, 
                                        unit_index=unit_index)
                
                sptr = SpikeTrain(np.array([]), units='s', copy=False, t_start=t_start, t_stop=t_stop)
                sptr.lazy_shape = (nb,)
            
            seg.spiketrains.append(sptr)
        
        # Events/Epoch
        event_channels = self.header['event_channels']
        #~ print('yep', event_channels)
        #~ exit()
        for chan_ind in range(len(event_channels)):
            if not lazy:
                ev_timestamp, ev_durations, ev_labels = self.event_timestamps(block_index=block_index, seg_index=seg_index, 
                                        event_channel_index=chan_ind)
                ev_times = self.rescale_event_timestamp(ev_timestamp, 'float64')
                ev_labels = ev_labels.astype('S')
            else:
                nb = self.event_count(block_index=block_index, seg_index=seg_index, 
                                        event_channel_index=chan_ind)
                lazy_shape = (nb,)
                ev_times = np.array([])
                ev_labels = np.array([], dtype='S')
                ev_durations = np.array([])
                
            name = event_channels['name'][chan_ind]
            if event_channels['type'][chan_ind] == b'event':
                e = Event(times=ev_times, labels=ev_labels, name=name, units='s', copy=False)
                e.segment = seg
                seg.events.append(e)
            elif event_channels['type'][chan_ind] == b'epoch':
                e = Epoch(times=ev_times, durations=ev_durations, labels=ev_labels, name=name, 
                                        units='s', copy=False)
                e.segment = seg
                seg.epochs.append(e)
            
            if lazy:
                e.lazy_shape = lazy_shape
        
        return seg


    def _make_signal_channel_groups(self, signal_group_mode='group-by-same-units'):
        
        channels = self.header['signal_channels']
        groups = collections.OrderedDict()
        if signal_group_mode=='group-by-same-units':
            all_units = np.unique(channels['units'])

            for i, unit in enumerate(all_units):
                ind, = np.nonzero(channels['units']==unit)
                groups[i] = ind

        elif signal_group_mode=='split-all':
            for i in range(channels.size):
                groups[i] = np.array([i])
        else:
            raise(NotImplementedError)
        return groups
        
