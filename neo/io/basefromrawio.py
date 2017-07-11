# -*- coding: utf-8 -*-
"""
BaseFromRaw
======

BaseFromRaw implement a bridge between the new neo.rawio API
and the neo.io legacy that give neo.core object.
The neo.rawio API is more restricted and limited and do not cover tricky
cases with asymetrical tree of neo object.
But if a format is done in neo.rawio the neo.io is done for free 
by inheritance of this class.


"""
# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

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
    
    def read_block(self, block_index=0, lazy=False, cascade=True, signal_group_mode=None,  
                    load_waveforms=False):
        
        if signal_group_mode is None:
            signal_group_mode = self.__prefered_signal_group_mode

        #annotations
        bl_annotations = dict(self.raw_annotations['blocks'][block_index])
        bl_annotations.pop('segments')

        bl = Block(**bl_annotations)
        
        if not cascade:
            return bl
        
        #ChannelIndex are plit in 2 parts:
        #  * some for AnalogSignals
        #  * some for Units
        
        #ChannelIndex ofr AnalogSignals
        channels = self.header['signal_channels']
        for i, ind in self._make_signal_channel_groups(signal_group_mode=signal_group_mode).items():
            channel_index = ChannelIndex(index=ind, channel_names=channels[ind]['name'].astype('S'),
                            channel_ids=channels[ind]['id'], name='Channel group {}'.format(i))
            bl.channel_indexes.append(channel_index)
        
        #ChannelIndex and Unit
        #TODO find something better than this
        # For simplification Unit are not attached to real channelindex
        # but a new ChannelIndex is created for that!!!!
        unit_channels = self.header['unit_channels']
        for c in range(len(unit_channels)):
            unit = Unit(name=unit_channels['name'][c], 
                            id=unit_channels['id'][c])
            channel_index = ChannelIndex(index=np.array([-2], dtype='i'),
                                    name='ChannelIndex for Unit')
            channel_index.units.append(unit)
            bl.channel_indexes.append(channel_index)
        
        #Segment
        for seg_index in range(self.segment_count(block_index)):
            seg =  self.read_segment(block_index=block_index, seg_index=seg_index, 
                                                                lazy=lazy, cascade=cascade, signal_group_mode=signal_group_mode,
                                                                load_waveforms=load_waveforms)
            bl.segments.append(seg)
            
            for i, anasig in enumerate(seg.analogsignals):
                bl.channel_indexes[i].analogsignals.append(anasig)
        
        bl.create_many_to_one_relationship()
        
        return bl

    def read_segment(self, block_index=0, seg_index=0, lazy=False, cascade=True, 
                        signal_group_mode=None, load_waveforms=False):

        if signal_group_mode is None:
            signal_group_mode = self.__prefered_signal_group_mode

        #annotations
        seg_annotations = dict(self.raw_annotations['blocks'][block_index]['segments'][seg_index])
        for k in ('signals', 'units', 'events'):
            seg_annotations.pop(k)
        
        seg = Segment(index=seg_index, **seg_annotations)#name,

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
            
            if signal_group_mode=='split-all':
                #in that case annotations by channel is OK
                chan_index = ind[0]
                d = self.raw_annotations['blocks'][block_index]['segments'][seg_index]['signals'][chan_index]
                annotations = dict(d)
                if 'name' not in annotations:
                    annotations['name'] = signal_channels['name'][chan_index]
            else:
                # when channel are grouped by same unit
                # annotations are empty...
                annotations = {}

            if lazy:
                anasig = AnalogSignal(np.array([]), units=units,  copy=False,
                        sampling_rate=sr, t_start=t_start, **annotations)
                anasig.lazy_shape = (sig_shape[0], len(ind))
            else:
                anasig = AnalogSignal(float_signal[:, ind], units=units,  copy=False,
                        sampling_rate=sr, t_start=t_start, **annotations)
            
            seg.analogsignals.append(anasig)
        
        #SpikeTrain and waveforms (optional)
        unit_channels = self.header['unit_channels']
        for unit_index in range(len(unit_channels)):
            if not lazy and load_waveforms:
                raw_waveforms = self.spike_raw_waveforms(block_index=block_index, seg_index=seg_index, 
                                                    unit_index=unit_index, t_start=None, t_stop=None)
                float_waveforms = self.rescale_waveforms_to_float(raw_waveforms, dtype='float32',
                                unit_index=unit_index)
                wf_units = unit_channels['wf_units'][unit_index]
                waveforms = pq.Quantity(float_waveforms, units=wf_units, dtype='float32', copy=False)
                wf_sampling_rate = unit_channels['wf_sampling_rate'][unit_index]
                wf_left_sweep = unit_channels['wf_left_sweep'][unit_index]
                wf_sampling_rate = wf_sampling_rate*pq.Hz
                wf_left_sweep = float(wf_left_sweep)/wf_sampling_rate * pq.s
            else:
                waveforms = None
                wf_left_sweep = None
                wf_sampling_rate = None                 
            
            d = self.raw_annotations['blocks'][block_index]['segments'][seg_index]['units'][unit_index]
            annotations = dict(d)
            if 'name' not in annotations:
                annotations['name'] = unit_channels['name'][c]
            
            if not lazy:
                spike_timestamp = self.spike_timestamps(block_index=block_index, seg_index=seg_index, 
                                        unit_index=unit_index, t_start=None, t_stop=None)
                spike_times = self.rescale_spike_timestamp(spike_timestamp, 'float64')
                
                sptr = SpikeTrain(spike_times, units='s', copy=False, t_start=t_start, t_stop=t_stop,
                                waveforms=waveforms, left_sweep=wf_left_sweep, 
                                sampling_rate=wf_sampling_rate, **annotations)
            else:
                nb = self.spike_count(block_index=block_index, seg_index=seg_index, 
                                        unit_index=unit_index)
                
                sptr = SpikeTrain(np.array([]), units='s', copy=False, t_start=t_start,
                                t_stop=t_stop, **annotations)
                sptr.lazy_shape = (nb,)
            
            
            seg.spiketrains.append(sptr)
        
        # Events/Epoch
        event_channels = self.header['event_channels']
        for chan_ind in range(len(event_channels)):
            if not lazy:
                ev_timestamp, ev_raw_durations, ev_labels = self.event_timestamps(block_index=block_index, seg_index=seg_index, 
                                        event_channel_index=chan_ind)
                ev_times = self.rescale_event_timestamp(ev_timestamp, 'float64') * pq.s
                if ev_raw_durations is None:
                    ev_durations = None
                else:
                    ev_durations = self.rescale_epoch_duration(ev_raw_durations, 'float64') * pq.s
                ev_labels = ev_labels.astype('S')
            else:
                nb = self.event_count(block_index=block_index, seg_index=seg_index, 
                                        event_channel_index=chan_ind)
                lazy_shape = (nb,)
                ev_times = np.array([]) * pq.s
                ev_labels = np.array([], dtype='S')
                ev_durations = np.array([]) * pq.s
            
            d = self.raw_annotations['blocks'][block_index]['segments'][seg_index]['events'][chan_ind]
            annotations = dict(d)
            if 'name' not in annotations:
                annotations['name'] = event_channels['name'][chan_ind]
            
            if event_channels['type'][chan_ind] == b'event':
                e = Event(times=ev_times, labels=ev_labels, units='s', copy=False, **annotations)
                e.segment = seg
                seg.events.append(e)
            elif event_channels['type'][chan_ind] == b'epoch':
                e = Epoch(times=ev_times, durations=ev_durations, labels=ev_labels,
                                        units='s', copy=False, **annotations)
                e.segment = seg
                seg.epochs.append(e)
            
            if lazy:
                e.lazy_shape = lazy_shape
        
        return seg


    def _make_signal_channel_groups(self, signal_group_mode='group-by-same-units'):
        """
        Aggregate signal channels with same units or split them all.
        """
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
        
