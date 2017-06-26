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

    supported_objects = [Block, Segment, AnalogSignal, ] #ChannelIndex, SpikeTrain, Unit, Event, 
    readable_objects = [Block, Segment]
    writeable_objects = []

    is_streameable = True

    name = 'BaseIO'
    description = ''
    extentions = []

    mode = 'file'

    def __init__(self, **kargs):
        BaseIO.__init__(self, **kargs)
        self.parse_header()
    
    def read_all_blocks(self, **kargs):
        blocks = []
        for bl_index in range(self.block_count()):
            bl = self.read_block(block_index=bl_index, **kargs)
            blocks.append(bl)
        return blocks
    
    def read_block(self, block_index=0, lazy=False, cascade=True, group_mode='group-by-units',  **kargs):
        
        bl = Block(name='Block {}'.format(block_index))
        if not cascade:
            return bl
        
        channels = self.header['signal_channels']
        for i, ind in self._make_channel_groups(group_mode=group_mode).items():
            channel_index = ChannelIndex(index=ind, channel_names=channels[ind]['name'],
                            channel_ids=channels[ind]['id'], name='Channel group {}'.format(i))
            bl.channel_indexes.append(channel_index)
        
        for seg_index in range(self.segment_count(block_index)):
            seg =  self.read_segment(block_index=block_index, seg_index=seg_index, 
                                                                lazy=lazy, cascade=cascade, group_mode=group_mode, **kargs)
            seg = Segment(index=seg_index)
            bl.segments.append(seg)
            
            for i, anasig in enumerate(seg.analogsignals):
                bl.channel_indexes[i].analogsignals.append(anasig)
        
        bl.create_many_to_one_relationship()
        
        return bl

    def read_segment(self, block_index=0, seg_index=0, lazy=False, cascade=True, 
                        group_mode='group-by-units', **kargs):
        seg = Segment(index=seg_index)#name, 
        
        if not cascade:
            return seg
        
        channels = self.header['signal_channels']
        
        channel_indexes=np.arange(channels.size)
        raw_signal = self.get_analogsignal_chunk(block_index=block_index, seg_index=seg_index,
                        i_start=None, i_stop=None, channel_indexes=channel_indexes)
        
        if not lazy:
            float_signal = self.rescale_raw_to_float(raw_signal,  dtype='float32', channel_indexes=channel_indexes)
        
        sample_rate = 0
        t_start = 0
        for i, ind in self._make_channel_groups(group_mode=group_mode).items():
            units = np.unique(channels[ind]['units'])
            print(units)
            assert len(units)==1
            units = units[0]
            
            if lazy:
                anasig = AnalogSignal(np.array([]), units=units,  copy=False,
                        sampling_rate=sample_rate*pq.Hz,t_start=t_start*pq.s)
                anasig.lay_shape = (raw_signal.shape[0], len(ind))
            else:
                anasig = AnalogSignal(float_signal[:, ind], units=units,  copy=False,
                        sampling_rate=sample_rate*pq.Hz,t_start=t_start*pq.s)
            
            seg.analogsignals.append(anasig)
        
        return seg
    
    def _make_channel_groups(self, group_mode='group-by-units'):
        
        channels = self.header['signal_channels']
        groups = collections.OrderedDict()
        if group_mode=='group-by-units':
            all_units = np.unique(channels['units'])
            print('all_units', all_units)
            for i, unit in enumerate(all_units):
                ind, = np.nonzero(channels['units']==unit)
                groups[i] = ind
                print(i, unit, ind)
        elif group_mode=='split-all':
            for i in range(channels.size):
                groups[i] = np.array([i])
        else:
            raise(NotImplementedError)
        return groups
        
