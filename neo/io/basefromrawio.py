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
from __future__ import print_function, division, absolute_import
#from __future__ import unicode_literals is not compatible with numpy.dtype both py2 py3

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
    """
    This implement generic reader on top of RawIO reader.
    
    Arguments depend on `mode` (dir or file)
    
    File case::
        
        reader = BlackRockIO(filename='FileSpec2.3001.nev')
    
    Dir case::
        
        reader = NeuralynxIO(dirname='Cheetah_v5.7.4/original_data')
    
    Other arguments are IO specific.

    """
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
    
    _prefered_signal_group_mode = 'split-all' #'group-by-same-units'
    _prefered_units_group_mode = 'split-all' # 'all-in-one'
    

    def __init__(self, *args, **kargs):
        BaseIO.__init__(self, *args, **kargs)
        self.parse_header()
    
    def read_block(self, block_index=0, lazy=False, cascade=True, signal_group_mode=None, 
                units_group_mode=None, load_waveforms=False, time_slices=None):
        """
        
        
        :param block_index: int default 0. In case of several block block_index can be specified.
        
        :param lazy: False by default. 
        
        :param cascade: True by Default
        
        :param signal_group_mode: 'split-all' or 'group-by-same-units' (default depend IO):
        This control behavior for grouping channels in AnalogSignal.
            * 'split-all': each channel will give an AnalogSignal
            * 'group-by-same-units' all channel sharing the same quantity units ar grouped in
            a 2D AnalogSignal
        
        :param units_group_mode: 'split-all' or 'all-in-one'(default depend IO)
        This control behavior for grouping Unit in ChannelIndex:
            * 'split-all': each neo.Unit is assigned to a new neo.ChannelIndex
            * 'all-in-one': all neo.Unit are grouped in the same neo.ChannelIndex (global spike sorting for instance)
        
        :param load_waveforms: False by default. Control SpikeTrains.waveforms is None or not.
        
        :param time_slices: None by default. List of time_slice. A time slice is (t_start, t_stop) both are quantities.
            each element will lead to a fake neo.Segment. So len(block.segment) == len(time_slice)
            all time_slice must be compatible with original time range.
        
        """
        
        if signal_group_mode is None:
            signal_group_mode = self._prefered_signal_group_mode

        if units_group_mode is None:
            units_group_mode = self._prefered_units_group_mode

        #annotations
        bl_annotations = dict(self.raw_annotations['blocks'][block_index])
        bl_annotations.pop('segments')
        bl_annotations = check_annotations(bl_annotations)

        bl = Block(**bl_annotations)
        
        if not cascade:
            return bl
        
        #ChannelIndex are plit in 2 parts:
        #  * some for AnalogSignals
        #  * some for Units
        
        #ChannelIndex ofr AnalogSignals
        all_channels = self.header['signal_channels']
        channel_indexes_list = self.get_group_channel_indexes()
        for channel_index in channel_indexes_list:
            for i, (ind_within, ind_abs) in self._make_signal_channel_subgroups(channel_index, 
                                                        signal_group_mode=signal_group_mode).items():
                neo_channel_index = ChannelIndex(index=ind_within, channel_names=all_channels[ind_abs]['name'].astype('S'),
                                channel_ids=all_channels[ind_abs]['id'], name='Channel group {}'.format(i))
                bl.channel_indexes.append(neo_channel_index)
        
        #ChannelIndex and Unit
        #2 case are possible in neo defifferent IO have choosen one or other:
        #  * All units are group in the same ChannelIndex and indexes are all channels : 'all-in-one'
        #  * Each units is assigned to one ChannelIndex : 'split-all'
        # This is kept for compatibility
        unit_channels = self.header['unit_channels']
        if units_group_mode=='all-in-one':
            if unit_channels.size>0:
                channel_index = ChannelIndex(index=np.array([], dtype='i'),
                                        name='ChannelIndex for all Unit')
                bl.channel_indexes.append(channel_index)
            for c in range(unit_channels.size):
                unit_annotations = self.raw_annotations['unit_channels'][c]
                unit = Unit(**unit_annotations)
                channel_index.units.append(unit)
                
        elif units_group_mode=='split-all':
            for c in range(len(unit_channels)):
                unit_annotations = self.raw_annotations['unit_channels'][c]
                unit = Unit(**unit_annotations)
                channel_index = ChannelIndex(index=np.array([], dtype='i'),
                                        name='ChannelIndex for Unit')
                channel_index.units.append(unit)
                bl.channel_indexes.append(channel_index)
        
        if time_slices is None:
            #Read the real segment counts
            for seg_index in range(self.segment_count(block_index)):
                seg =  self.read_segment(block_index=block_index, seg_index=seg_index, 
                                                                    lazy=lazy, cascade=cascade, signal_group_mode=signal_group_mode,
                                                                    load_waveforms=load_waveforms)
                bl.segments.append(seg)
                
        else:
            #return a fake segment list corresponding to time_slices
            for s, time_slice in enumerate(time_slices):
                #find in which segment time_slice is
                t_start, t_stop = time_slice
                t_start = ensure_second(t_start)
                t_stop = ensure_second(t_stop)
                related_seg_index = None
                for seg_index in range(self.segment_count(block_index)):
                    seg_t_start = self.segment_t_start(block_index, seg_index) * pq.s
                    seg_t_stop = self.segment_t_stop(block_index, seg_index) * pq.s
                    if (seg_t_start<=t_start<=seg_t_stop) and (seg_t_start<=t_stop<=seg_t_stop):
                        related_seg_index = seg_index
                
                if related_seg_index is None:
                    raise(ValueError('time_slice not in any segment range  {}'.format(time_slice)))
                
                seg =  self.read_segment(block_index=block_index, seg_index=related_seg_index,
                                                                    lazy=lazy, cascade=cascade, signal_group_mode=signal_group_mode,
                                                                    load_waveforms=load_waveforms, time_slice=time_slice)
                seg.index = s
                bl.segments.append(seg)
                
                for c, anasig in enumerate(seg.analogsignals):
                    bl.channel_indexes[c].analogsignals.append(anasig)
        
        #create link to other containers ChannelIndex and Units
        for seg in bl.segments:
            for c, anasig in enumerate(seg.analogsignals):
                bl.channel_indexes[c].analogsignals.append(anasig)
            
            nsig = len(seg.analogsignals)
            for c, sptr in enumerate(seg.spiketrains):
                if units_group_mode=='all-in-one':
                    bl.channel_indexes[nsig].units[c].spiketrains.append(sptr)
                elif units_group_mode=='split-all':
                    bl.channel_indexes[nsig+c].units[0].spiketrains.append(sptr)
        
        bl.create_many_to_one_relationship()
        
        return bl

    def read_segment(self, block_index=0, seg_index=0, lazy=False, cascade=True, 
                        signal_group_mode=None, load_waveforms=False, time_slice=None):
        """
        :param block_index: int default 0. In case of several block block_index can be specified.
        
        :param seg_index: int default 0. Index of segment.
        
        :param lazy: False by default. 
        
        :param cascade: True by Default

        :param signal_group_mode: 'split-all' or 'group-by-same-units' (default depend IO):
        This control behavior for grouping channels in AnalogSignal.
            * 'split-all': each channel will give an AnalogSignal
            * 'group-by-same-units' all channel sharing the same quantity units ar grouped in
            a 2D AnalogSignal
        
        :param load_waveforms: False by default. Control SpikeTrains.waveforms is None or not.
        
        :param time_slice: None by default means no limit.
            A time slice is (t_start, t_stop) both are quantities.
            All object AnalogSignal, SpikeTrain, Event, Epoch will load only in the slice.
        """

        if signal_group_mode is None:
            signal_group_mode = self._prefered_signal_group_mode

        #annotations
        seg_annotations = dict(self.raw_annotations['blocks'][block_index]['segments'][seg_index])
        for k in ('signals', 'units', 'events'):
            seg_annotations.pop(k)
        seg_annotations = check_annotations(seg_annotations)
        
        seg = Segment(index=seg_index, **seg_annotations)

        if not cascade:
            return seg
        
        
        seg_t_start = self.segment_t_start(block_index, seg_index) * pq.s
        seg_t_stop = self.segment_t_stop(block_index, seg_index) * pq.s
        
        # get only a slice of objects limited by t_start and t_stop time_slice = (t_start, t_stop)
        if time_slice is None:
            t_start, t_stop = None, None
            t_start_, t_stop_ = None, None
        else:
            assert not lazy, 'time slice only work when not lazy'
            t_start, t_stop = time_slice
            
            t_start = ensure_second(t_start)
            t_stop = ensure_second(t_stop)
            
            #checks limits
            if t_start<seg_t_start:
                t_start = seg_t_start
            if t_stop>seg_t_stop:
                t_stop = seg_t_stop
            
            #in float format in second (for rawio clip)
            t_start_, t_stop_ = float(t_start.magnitude), float(t_stop.magnitude)
            
            #new spiketrain limits
            seg_t_start = t_start
            seg_t_stop = t_stop
            
        
        #AnalogSignal
        signal_channels = self.header['signal_channels']
        
        if signal_channels.size>0:
            channel_indexes_list = self.get_group_channel_indexes()
            for channel_indexes in channel_indexes_list:
                sr = self.get_signal_sampling_rate(channel_indexes) * pq.Hz
                sig_t_start = self.get_signal_t_start(block_index, seg_index, channel_indexes) * pq.s

                sig_size = self.get_signal_size(block_index=block_index, seg_index=seg_index, 
                                                                                        channel_indexes=channel_indexes)
                if not lazy:
                    #in case of time_slice get: get i_start, i_stop, new sig_t_start
                    if t_stop is not None:
                        i_stop = int((t_stop-sig_t_start).magnitude * sr.magnitude)
                        if i_stop>sig_size:
                            i_stop = sig_size
                    else:
                        i_stop = None
                    if t_start is not None:
                        i_start = int((t_start-sig_t_start).magnitude * sr.magnitude)
                        if i_start<0:
                            i_start = 0
                        sig_t_start += (i_start/sr).rescale('s')
                    else:
                        i_start = None
                    
                    raw_signal = self.get_analogsignal_chunk(block_index=block_index, seg_index=seg_index,
                                i_start=i_start, i_stop=i_stop, channel_indexes=channel_indexes)
                    float_signal = self.rescale_signal_raw_to_float(raw_signal,  dtype='float32',
                                                                                            channel_indexes=channel_indexes)
                
                for i, (ind_within, ind_abs) in self._make_signal_channel_subgroups(channel_indexes,
                                                signal_group_mode=signal_group_mode).items():
                    units = np.unique(signal_channels[ind_abs]['units'])
                    assert len(units)==1
                    units = ensure_signal_units(units[0])
                    
                    if signal_group_mode=='split-all':
                        #in that case annotations by channel is OK
                        chan_index = ind_abs[0]
                        d = self.raw_annotations['blocks'][block_index]['segments'][seg_index]['signals'][chan_index]
                        annotations = dict(d)
                        if 'name' not in annotations:
                            annotations['name'] = signal_channels['name'][chan_index]
                    else:
                        # when channel are grouped by same unit
                        # annotations are empty...
                        annotations = {}
                        annotations['name'] = 'Channel bundle ({}) '.format(','.join(signal_channels[ind_abs]['name']))
                    annotations = check_annotations(annotations)
                    if lazy:
                        anasig = AnalogSignal(np.array([]), units=units,  copy=False,
                                sampling_rate=sr, t_start=sig_t_start, **annotations)
                        anasig.lazy_shape = (sig_size, len(ind_within))
                    else:
                        anasig = AnalogSignal(float_signal[:, ind_within], units=units,  copy=False,
                                sampling_rate=sr, t_start=sig_t_start, **annotations)
                    seg.analogsignals.append(anasig)

                    
        
        #SpikeTrain and waveforms (optional)
        unit_channels = self.header['unit_channels']
        for unit_index in range(len(unit_channels)):
            if not lazy and load_waveforms:
                raw_waveforms = self.get_spike_raw_waveforms(block_index=block_index, seg_index=seg_index, 
                                                    unit_index=unit_index, t_start=t_start_, t_stop=t_stop_)
                float_waveforms = self.rescale_waveforms_to_float(raw_waveforms, dtype='float32',
                                unit_index=unit_index)
                wf_units = ensure_signal_units(unit_channels['wf_units'][unit_index])
                waveforms = pq.Quantity(float_waveforms, units=wf_units, dtype='float32', copy=False)
                wf_sampling_rate = unit_channels['wf_sampling_rate'][unit_index]
                wf_left_sweep = unit_channels['wf_left_sweep'][unit_index]
                if wf_left_sweep>0:
                    wf_left_sweep = float(wf_left_sweep)/wf_sampling_rate * pq.s
                else:
                    wf_left_sweep = None
                wf_sampling_rate = wf_sampling_rate*pq.Hz
            else:
                waveforms = None
                wf_left_sweep = None
                wf_sampling_rate = None                 
            
            d = self.raw_annotations['blocks'][block_index]['segments'][seg_index]['units'][unit_index]
            annotations = dict(d)
            if 'name' not in annotations:
                annotations['name'] = unit_channels['name'][c]
            annotations = check_annotations(annotations)
            
            if not lazy:
                spike_timestamp = self.get_spike_timestamps(block_index=block_index, seg_index=seg_index, 
                                        unit_index=unit_index, t_start=t_start_, t_stop=t_stop_)
                spike_times = self.rescale_spike_timestamp(spike_timestamp, 'float64')
                sptr = SpikeTrain(spike_times, units='s', copy=False, t_start=seg_t_start, t_stop=seg_t_stop,
                                    waveforms=waveforms, left_sweep=wf_left_sweep, 
                                    sampling_rate=wf_sampling_rate, **annotations)
            else:
                nb = self.spike_count(block_index=block_index, seg_index=seg_index, 
                                        unit_index=unit_index)
                sptr = SpikeTrain(np.array([]), units='s', copy=False, t_start=seg_t_start,
                                t_stop=seg_t_stop, **annotations)
                sptr.lazy_shape = (nb,)
            
            seg.spiketrains.append(sptr)
        
        # Events/Epoch
        event_channels = self.header['event_channels']
        for chan_ind in range(len(event_channels)):
            if not lazy:
                ev_timestamp, ev_raw_durations, ev_labels = self.get_event_timestamps(block_index=block_index, seg_index=seg_index, 
                                        event_channel_index=chan_ind, t_start=t_start_, t_stop=t_stop_)
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
            
            annotations = check_annotations(annotations)
            
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
        
        seg.create_many_to_one_relationship()
        return seg


    def _make_signal_channel_subgroups(self, channel_indexes, signal_group_mode='group-by-same-units'):
        """
        For some RawIO channel are already splitted in groups.
        But in any cases, channel need to be splitted again in sub groups
        because they do not have the same units.
        
        They can also be splitted one by one to match previous behavior for
        some IOs in older version of neo (<=0.5).
        
        This method aggregate signal channels with same units or split them all.
        """
        all_channels = self.header['signal_channels']
        if channel_indexes is None:
            channel_indexes = np.arange(all_channels.size, dtype=int)
        channels = all_channels[channel_indexes]
        
        groups = collections.OrderedDict()
        if signal_group_mode=='group-by-same-units':
            all_units = np.unique(channels['units'])

            for i, unit in enumerate(all_units):
                ind_within, = np.nonzero(channels['units']==unit)
                ind_abs = channel_indexes[ind_within]
                groups[i] = (ind_within, ind_abs)

        elif signal_group_mode=='split-all':
            for i, chan_index in enumerate(channel_indexes):
                ind_within = [i]
                ind_abs = channel_indexes[ind_within]
                groups[i] = (ind_within, ind_abs)
        else:
            raise(NotImplementedError)
        return groups


unit_convert = {'Volts': 'V',  'volts': 'V','Volt': 'V', 'volt': 'V', ' Volt' : 'V','microV': 'V'}
def ensure_signal_units(units):
    #test units
    units = units.replace(' ', '')
    if units in unit_convert:
        units = unit_convert[units]
    try:
        units = pq.Quantity(1, units)
    except:
        logging.warning('Units "{}" not understand use dimentionless instead'.format(units))
        units = ''
    return units

def check_annotations(annotations):
    #force type to str for some keys
    # imposed for tests
    for k in ('name', 'description', 'file_origin'):
        if k in annotations:
            annotations[k] = str(annotations[k])
    return annotations

def ensure_second(v):
    if isinstance(v, float):
        return v*pq.s
    elif isinstance(v, pq.Quantity):
        return v.rescale('s')
    elif isinstance(v, int):
        return float(v)*pq.s

