# -*- coding: utf-8 -*-
"""
Here a list for testing neo.rawio API compliance.
This is called automatically by `BaseTestRawIO`

All rules are listed as function so it should be easier to:
  * identify the rawio API
  * debug
  * discuss rules

"""
import time
if not hasattr(time, 'perf_counter'):
    time.perf_counter = time.time
import logging

import numpy as np



from neo.rawio.baserawio import (_signal_channel_dtype,_unit_channel_dtype, 
            _event_channel_dtype, _common_sig_characteristics)



def print_class(reader):
    return reader.__class__.__name__

def header_is_total(reader):
    """
    Test if hedaer contains:
      * 'signal_channels'
      * 'unit_channels'
      * 'event_channels'
    
    """
    h = reader.header
    
    assert 'signal_channels' in h, 'signal_channels missing in header'
    if h['signal_channels'] is not None:
        dt = h['signal_channels'].dtype
        for k, _ in _signal_channel_dtype:
            assert k in dt.fields, '%s not in signal_channels.dtype'%k

    assert 'unit_channels' in h, 'unit_channels missing in header'
    if h['unit_channels'] is not None:
        dt = h['unit_channels'].dtype
        for k, _ in _unit_channel_dtype:
            assert k in dt.fields, '%s not in unit_channels.dtype'%k
    
    assert 'event_channels' in h, 'event_channels missing in header'
    if h['event_channels'] is not None:
        dt = h['event_channels'].dtype
        for k, _ in _event_channel_dtype:
            assert k in dt.fields, '%s not in event_channels.dtype'%k


def count_element(reader):
    """
    Count block/segment/signals/spike/events
    
    """
    
    nb_sig = reader.signal_channels_count()
    nb_unit = reader.unit_channels_count()
    nb_event_channel = reader.event_channels_count()
    
    nb_block = reader.block_count()
    assert nb_block>0, '{} have {} block'.format(print_class(reader), nb_block)
    
    for block_index in range(nb_block):
        nb_seg = reader.segment_count(block_index)
        
        for seg_index in range(nb_seg):
            t_start = reader.segment_t_start(block_index=block_index, seg_index=seg_index)
            t_stop = reader.segment_t_stop(block_index=block_index, seg_index=seg_index)
            assert t_stop>t_start
            
            if nb_sig>0:
                if reader._several_channel_groups:
                    channel_indexes_list = reader.get_group_channel_indexes()
                    for channel_indexes in channel_indexes_list:
                        sig_size = reader.get_signal_size(block_index, seg_index,
                                                channel_indexes=channel_indexes)
                else:
                    sig_size = reader.get_signal_size(block_index, seg_index,
                                            channel_indexes=None)
                
                for unit_index in range(nb_unit):
                    nb_spike = reader.spike_count(block_index=block_index, seg_index=seg_index,
                                                        unit_index=unit_index)
                
                for event_channel_index in range(nb_event_channel):
                    nb_event = reader.event_count(block_index=block_index, seg_index=seg_index,
                                                        event_channel_index=event_channel_index)


def iter_over_sig_chunks(reader, channel_indexes, chunksize=1024):
    if channel_indexes is None:
        nb_sig = reader.signal_channels_count()
    else:
        nb_sig = len(channel_indexes)
    if nb_sig==0: return

    nb_block = reader.block_count()
    
    #read all chunk in RAW data
    chunksize = 1024
    for block_index in range(nb_block):
        nb_seg = reader.segment_count(block_index)
        for seg_index in range(nb_seg):
            sig_size = reader.get_signal_size(block_index, seg_index, channel_indexes)
            
            nb = sig_size//chunksize + 1
            for i in range(nb):
                i_start = i*chunksize
                i_stop = min((i+1)*chunksize, sig_size)
                raw_chunk = reader.get_analogsignal_chunk(block_index=block_index, seg_index=seg_index,
                            i_start=i_start, i_stop=i_stop, channel_indexes=channel_indexes)
                yield raw_chunk


def read_analogsignals(reader):
    """
    Read and convert some signals chunks.
    
    Test special case when signal_channels do not have same sampling_rate.
    AKA _need_chan_index_check
    """
    nb_sig = reader.signal_channels_count()
    if nb_sig==0: return
    
    if reader._several_channel_groups:
        channel_indexes_list = reader.get_group_channel_indexes()
    else:
        channel_indexes_list = [ None ]
    
    #read all chunk for all channel all block all segment
    for channel_indexes in channel_indexes_list:
        for raw_chunk in iter_over_sig_chunks(reader, channel_indexes, chunksize=1024):
            assert raw_chunk.ndim ==2
            #~ pass
    
    for channel_indexes in channel_indexes_list:
        sr = reader.get_signal_sampling_rate(channel_indexes=channel_indexes)
        assert type(sr) == float, 'Type of sampling is {} should float'.format(type(sr))
        
        
    # make other test on the first chunk of first block first block
    block_index=0
    seg_index=0
    for channel_indexes in channel_indexes_list:
        i_start = 0
        sig_size = reader.get_signal_size(block_index, seg_index,
                                                channel_indexes=channel_indexes)
        i_stop = min(1024, sig_size)
        
        if channel_indexes is None:
            nb_sig = reader.header['signal_channels'].size
            channel_indexes = np.arange(nb_sig, dtype=int)
        
        all_signal_channels = reader.header['signal_channels']
        
        signal_names = all_signal_channels['name'][channel_indexes]
        signal_ids = all_signal_channels['id'][channel_indexes]
        
        unique_chan_name = (np.unique(signal_names).size == all_signal_channels.size)
        unique_chan_id = (np.unique(signal_ids).size == all_signal_channels.size)
        
        # acces by channel inde/ids/names should give the same chunk
        channel_indexes2 = channel_indexes[::2]
        channel_names2 = signal_names[::2]
        channel_ids2 = signal_ids[::2]
        
        raw_chunk0 = reader.get_analogsignal_chunk(block_index=block_index, seg_index=seg_index,
                                i_start=i_start, i_stop=i_stop,  channel_indexes=channel_indexes2)
        assert raw_chunk0.ndim==2
        assert raw_chunk0.shape[0]==i_stop
        assert raw_chunk0.shape[1]==len(channel_indexes2)
        
        if unique_chan_name:
            raw_chunk1 = reader.get_analogsignal_chunk(block_index=block_index, seg_index=seg_index,
                                    i_start=i_start, i_stop=i_stop,  channel_names=channel_names2)
            np.testing.assert_array_equal(raw_chunk0, raw_chunk1)
        
        if unique_chan_id:
            raw_chunk2 = reader.get_analogsignal_chunk(block_index=block_index, seg_index=seg_index,
                                    i_start=i_start, i_stop=i_stop,  channel_ids=channel_ids2)
            np.testing.assert_array_equal(raw_chunk0, raw_chunk2)
    
        #convert to float32/float64
        for dt in ('float32', 'float64'):
            float_chunk0 = reader.rescale_signal_raw_to_float(raw_chunk0, dtype=dt,
                            channel_indexes=channel_indexes2)
            if unique_chan_name:
                float_chunk1 = reader.rescale_signal_raw_to_float(raw_chunk1, dtype=dt,
                                channel_names=channel_names2)
            if unique_chan_id:
                float_chunk2 = reader.rescale_signal_raw_to_float(raw_chunk2, dtype=dt,
                            channel_ids=channel_ids2)
                            
            assert float_chunk0.dtype==dt
            if unique_chan_name:
                np.testing.assert_array_equal(float_chunk0, float_chunk1)
            if unique_chan_id:
                np.testing.assert_array_equal(float_chunk0, float_chunk2)

def benchmark_speed_read_signals(reader):
    """
    A very basic speed measurement that read all signal
    in a file.
    """
    
    if reader._several_channel_groups:
        channel_indexes_list = reader.get_group_channel_indexes()
    else:
        channel_indexes_list = [ None ]
    
    for channel_indexes in channel_indexes_list:
        if channel_indexes is None:
            nb_sig = reader.signal_channels_count()
        else:
            nb_sig = len(channel_indexes)
        if nb_sig==0: continue
        
        nb_samples = 0
        t0 = time.perf_counter()
        for raw_chunk in iter_over_sig_chunks(reader, channel_indexes, chunksize=1024):
            nb_samples += raw_chunk.shape[0]
        t1 = time.perf_counter()
        speed = (nb_samples*nb_sig)/(t1-t0)/1e6
        logging.info('{} read ({}signals x {}samples) in {:0.3f} s so speed {:0.3f} MSPS from {}'.format(print_class(reader),
                                    nb_sig, nb_samples, t1-t0, speed, reader.source_name()))



def read_spike_times(reader):
    """
    Read and convert all spike times.
    """
    
    nb_block = reader.block_count()
    nb_unit = reader.unit_channels_count()
    
    for block_index in range(nb_block):
        nb_seg = reader.segment_count(block_index)
        for seg_index in range(nb_seg):
            for unit_index in range(nb_unit):
                nb_spike =  reader.spike_count(block_index=block_index,
                                        seg_index=seg_index, unit_index=unit_index)
                if nb_spike==0: continue
                
                spike_timestamp = reader.get_spike_timestamps(block_index=block_index, seg_index=seg_index,
                                                    unit_index=unit_index, t_start=None, t_stop=None)
                assert spike_timestamp.shape[0] == nb_spike, 'nb_spike {} != {}'.format(spike_timestamp.shape[0] , nb_spike)
                
                spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
                assert spike_times.dtype=='float64'
                
                if spike_times.size>3:
                    #load only one spike by forcing limits
                    t_start = spike_times[1] - 0.001
                    t_stop = spike_times[1] + 0.001
                    
                    spike_timestamp2 = reader.get_spike_timestamps(block_index=block_index, seg_index=seg_index,
                                                    unit_index=unit_index, t_start=t_start, t_stop=t_stop)
                    assert spike_timestamp2.shape[0]==1
                    
                    spike_times2 = reader.rescale_spike_timestamp(spike_timestamp2, 'float64')
                    assert spike_times2[0] == spike_times[1]
                
                


def read_spike_waveforms(reader):
    """
    Read and convert some all waveforms.
    """
    nb_block = reader.block_count()
    nb_unit = reader.unit_channels_count()
    
    for block_index in range(nb_block):
        nb_seg = reader.segment_count(block_index)
        for seg_index in range(nb_seg):
            for unit_index in range(nb_unit):
                nb_spike =  reader.spike_count(block_index=block_index,
                                        seg_index=seg_index, unit_index=unit_index)
                if nb_spike==0: continue
                
                raw_waveforms = reader.get_spike_raw_waveforms(block_index=block_index, 
                                                    seg_index=seg_index, unit_index=unit_index,
                                                    t_start=None, t_stop=None)
                if raw_waveforms is None:
                    continue
                assert raw_waveforms.shape[0] == nb_spike
                assert raw_waveforms.ndim == 3
                
                for dt in ('float32', 'float64'):
                    float_waveforms = reader.rescale_waveforms_to_float(raw_waveforms, dtype=dt, unit_index=unit_index)
                    assert float_waveforms.dtype==dt
                    assert float_waveforms.shape==raw_waveforms.shape



def read_events(reader):
    """
    Read and convert some event or epoch.
    """
    nb_block = reader.block_count()
    nb_event_channel = reader.event_channels_count()
    
    for block_index in range(nb_block):
        nb_seg = reader.segment_count(block_index)
        for seg_index in range(nb_seg):
            for ev_chan in range(nb_event_channel):
                nb_event = reader.event_count(block_index=block_index, seg_index=seg_index,
                                                                                    event_channel_index=ev_chan)
                if nb_event==0: continue

                ev_timestamps, ev_durations, ev_labels = reader.get_event_timestamps(block_index=block_index, seg_index=seg_index,
                                                event_channel_index=ev_chan)
                assert ev_timestamps.shape[0] == nb_event, 'Wrong shape {}, {}'.format(ev_timestamps.shape[0], nb_event)
                if ev_durations is not None:
                    assert ev_durations.shape[0] == nb_event
                assert ev_labels.shape[0] == nb_event
                
                ev_times = reader.rescale_event_timestamp(ev_timestamps, dtype='float64')
                assert ev_times.dtype=='float64'
                
def has_annotations(reader):
    assert hasattr(reader, 'raw_annotations'), 'raw_annotation are not set'
    

