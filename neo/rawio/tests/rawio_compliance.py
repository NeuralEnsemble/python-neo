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
import logging

import numpy as np



from neo.rawio.baserawio import (_signal_channel_dtype,_unit_channel_dtype, 
            _event_channel_dtype)



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
            if nb_sig>0:
                sig_shape = reader.analogsignal_shape(block_index, seg_index)
                
                for unit_index in range(nb_unit):
                    nb_spike = reader.spike_count(block_index=block_index, seg_index=seg_index,
                                                        unit_index=unit_index)
                
                for event_channel_index in range(nb_event_channel):
                    nb_event = reader.event_count(block_index=block_index, seg_index=seg_index,
                                                        event_channel_index=event_channel_index)


def iter_over_sig_chunks(reader, chunksize = 1024):
    nb_sig = reader.signal_channels_count()
    if nb_sig==0: return

    nb_block = reader.block_count()
    
    #read all chunk in RAW data
    chunksize = 1024
    channel_indexes = np.arange(nb_sig, dtype=int)
    for block_index in range(nb_block):
        nb_seg = reader.segment_count(block_index)
        for seg_index in range(nb_seg):
            sig_shape = reader.analogsignal_shape(block_index, seg_index)
            
            nb = sig_shape[0]//chunksize + 1
            for i in range(nb):
                i_start = i*chunksize
                i_stop = min((i+1)*chunksize, sig_shape[0])
                raw_chunk = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=i_start, i_stop=i_stop, channel_indexes=channel_indexes)
                yield raw_chunk


def read_analogsignals(reader):
    """
    Read and convert some signals chunks.
    
    
    """
    nb_sig = reader.signal_channels_count()
    if nb_sig==0: return
    
    #read all chunk for all channel all block all segmen
    for raw_chunk in iter_over_sig_chunks(reader, chunksize = 1024):
        pass
    
    
    # make other test on the first chunk
    i_start = 0
    sig_shape = reader.analogsignal_shape(0, 0)
    i_stop = min(1024, sig_shape[0])
    
    # acces by channel inde/ids/names should give the same chunk
    channel_indexes = np.arange(nb_sig, dtype=int)[::2]
    channel_names = reader.header['signal_channels']['name'][::2]
    channel_ids = reader.header['signal_channels']['id'][::2]
    
    raw_chunk0 = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=i_start, i_stop=i_stop,  channel_indexes=channel_indexes)
    raw_chunk1 = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=i_start, i_stop=i_stop,  channel_names=channel_names)
    raw_chunk2 = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=i_start, i_stop=i_stop,  channel_ids=channel_ids)
    np.testing.assert_array_equal(raw_chunk0, raw_chunk1)
    np.testing.assert_array_equal(raw_chunk0, raw_chunk2)
    
    
    #convert to float32/float64
    for dt in ('float32', 'float64'):
        float_chunk0 = reader.rescale_signal_raw_to_float(raw_chunk0, dtype=dt,
                        channel_indexes=channel_indexes)
        float_chunk1 = reader.rescale_signal_raw_to_float(raw_chunk1, dtype=dt,
                        channel_names=channel_names)
        float_chunk2 = reader.rescale_signal_raw_to_float(raw_chunk2, dtype=dt,
                        channel_ids=channel_ids)
                        
        assert float_chunk0.dtype==dt
        np.testing.assert_array_equal(float_chunk0, float_chunk1)
        np.testing.assert_array_equal(float_chunk0, float_chunk2)

def benchmark_speed_read_signals(reader):
    """
    A very basic speed measurement that read all signal
    in a file.
    """
    nb_sig = reader.signal_channels_count()
    if nb_sig==0: return
    
    t0 = time.perf_counter()
    for raw_chunk in iter_over_sig_chunks(reader, chunksize = 1024):
        pass
    t1 = time.perf_counter()
    logging.info('{} read signals of {} in {:0.3f}s'.format(print_class(reader), reader.source_name(), t1-t0))



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
                
                spike_timestamp = reader.spike_timestamps(block_index=block_index, 
                                                    unit_index=unit_index, t_start=None, t_stop=None)
                assert spike_timestamp.shape[0] == nb_spike
                
                spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
                assert spike_times.dtype=='float64'
                
                if spike_times.size>3:
                    t_start = spike_times[1]
                    t_stop = spike_times[1]
                    
                    spike_timestamp2 = reader.spike_timestamps(block_index=block_index, 
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
                
                raw_waveforms = reader.spike_raw_waveforms(block_index=block_index, 
                                                    unit_index=unit_index, t_start=None, t_stop=None)
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

                ev_timestamps, ev_durations, ev_labels = reader.event_timestamps(block_index=0, seg_index=0,
                                                event_channel_index=ev_chan)
                assert ev_timestamps.shape[0] == nb_event
                if ev_durations is not None:
                    assert ev_durations.shape[0] == nb_event
                assert ev_labels.shape[0] == nb_event
                
                ev_times = reader.rescale_event_timestamp(ev_timestamps, dtype='float64')
                assert ev_times.dtype=='float64'
                
                

