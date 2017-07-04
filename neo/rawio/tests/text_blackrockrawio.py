
import neo
#~ import neo.rawio.blackrockrawio
from  neo.rawio.blackrockrawio import BlackrockRawIO
from neo.io import BlackrockIO
import numpy as np

import time

def test_BlackrockRawIO():
    #~ filename = '/media/sgarcia/SamCNRS/DataSpikeSorting/elodie/Dataset 3/20161124-112218-001'
    #~ filename = '/media/sgarcia/SamCNRS/DataSpikeSorting/elodie/Nouveaux_datasets/micro_VS10_SAB 1 600ms/20160627-155334-001'
    #~ filename = '/media/sgarcia/SamCNRS/DataSpikeSorting/elodie/Nouveaux_datasets/micro_VS10_SAB 1 600ms/20160627-155334-001'
    #~ filename = '/home/samuel/Documents/projet/DataSpikeSorting/elodie/Dataset 3/20161124-112218-001.ns5'
    #~ filename = '/home/sgarcia/Documents/projet/tridesclous_examples/20160627-161211-001'
    filename = '/home/sgarcia/Documents/files_for_testing_neo/blackrock/FileSpec2.3001.ns5'
    
    
    reader = BlackrockRawIO(filename=filename, nsx_to_load=5)
    reader.parse_header()
    print(reader)
    
    assert reader.block_count()==1
    
    assert reader.segment_count(0)==1
    
    # Acces 10000 chunk of 1024 samples for 10 channels
    nb_chunk = 10000
    chunksize = 1024
    #~ channel_indexes = np.arange(20,30)
    channel_indexes = np.arange(5,10)
    
    t0 = time.perf_counter()
    for i in range(nb_chunk):
        i_start = i*chunksize
        i_stop = (i+1)*chunksize
        raw_chunk = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=i_start, i_stop=i_stop,  channel_indexes=channel_indexes)
        #~ print(sig_chunk.shape)
    t1 = time.perf_counter()
    print('acces {} chunk of {} samples in {:0.3f} s'.format(nb_chunk,chunksize, t1-t0))
    
    sig_shape = reader.analogsignal_shape(0,0)
    #~ print('sig_shape', sig_shape)
    
    channel_names = reader.header['signal_channels']['name'][[1,4,3,7]]
    #~ print(channel_names)
    
    raw_chunk = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=0, i_stop=1024,  channel_names=channel_names)
    
    float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, dtype='float64', channel_names=channel_names)
    
    
    #~ import matplotlib.pyplot as plt
    #~ fig, axs = plt.subplots(nrows=2, sharex=True)
    #~ axs[0].plot(raw_chunk)
    #~ axs[1].plot(float_chunk)
    #~ plt.show()
    print(reader.header['unit_channels'])
    nb_unit = reader.unit_channels_count()
    for unit_index in range(nb_unit):
        print()
        print(reader.header['unit_channels'][unit_index])
        
        nb =  reader.spike_count(unit_index=unit_index)
        print('nb', nb)
        spike_timestamp = reader.spike_timestamps(unit_index=unit_index, t_start=None, t_stop=None)
        print(spike_timestamp.shape)
        print(spike_timestamp[:10])
        spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
        print(spike_times[:10])
        
        raw_waveforms = reader.spike_raw_waveforms(block_index=0, seg_index=0, unit_index=unit_index, t_start=None, t_stop=None)
        print(raw_waveforms.shape, raw_waveforms.dtype)
        
        float_waveforms = reader.rescale_waveforms_to_float(raw_waveforms, dtype='float32', unit_index=unit_index)
        print(float_waveforms.shape, float_waveforms.dtype)
        
    
    
    print(reader.header['event_channels'])
    nb = reader.event_channels_count()
    for i in range(nb):
        nb_event = reader.event_count(block_index=0, seg_index=0, event_channel_index=i)
        print('i', i, 'nb_event', nb_event)
        ev_timestamps, ev_durations, ev_labels = reader.event_timestamps(block_index=0, seg_index=0, event_channel_index=i)
        print(ev_timestamps)
        print(ev_durations)
        print('ev_labels', ev_labels)
        ev_times = reader.rescale_event_timestamp(ev_timestamps, dtype='float64')
        print(ev_times)
        
    
    
    
    
    


def test_BlackrockIO():
    #~ filename = '/home/samuel/Documents/projet/DataSpikeSorting/elodie/Dataset 3/20161124-112218-001.ns5'
    #~ filename = '/home/sgarcia/Documents/projet/tridesclous_examples/20160627-161211-001'
    #~ filename = '/home/samuel/Téléchargements/files_for_testing_neo/blackrock/FileSpec2.3001.ns5'
    filename = '/home/sgarcia/Documents/files_for_testing_neo/blackrock/FileSpec2.3001.ns5'
    
    reader = BlackrockIO(filename=filename, nsx_to_load=5)
    
    #~ blocks = reader.read(lazy=True, signal_group_mode='group-by-same-units')
    blocks = reader.read(lazy=False, signal_group_mode='group-by-same-units')
    #~ blocks = reader.read(lazy=False, signal_group_mode='split-all')
    
    for bl in blocks:
        print()
        print(bl)
        for seg in bl.segments:
            print(seg)
            print(seg.analogsignals)
            for anasig in seg.analogsignals:
                print(anasig.name, anasig.shape)
            #~ print(len(seg.spiketrains))
            for sptr in seg.spiketrains:
                print(sptr.name, sptr.shape)
            
            for ev in seg.events:
                print(ev.name)
                #~ print(ev.labels)
                print('yep')
            
            for ep in seg.epochs:
                print(ep)

    
if __name__ == '__main__':
    test_BlackrockRawIO()
    #~ test_BlackrockIO()



