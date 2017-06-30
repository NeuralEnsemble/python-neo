
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
    filename = '/home/samuel/Téléchargements/files_for_testing_neo/blackrock/FileSpec2.3001.ns5'
    
    
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
        spike_timestamp = reader.spike_timestamps(unit_index=unit_index, ind_start=None, ind_stop=None)
        print(spike_timestamp.shape)
        print(spike_timestamp[:10])
        spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
        print(spike_times[:10])
    
    
    
    
    


def test_BlackrockIO():
    #~ filename = '/home/samuel/Documents/projet/DataSpikeSorting/elodie/Dataset 3/20161124-112218-001.ns5'
    #~ filename = '/home/sgarcia/Documents/projet/tridesclous_examples/20160627-161211-001'
    filename = '/home/samuel/Téléchargements/files_for_testing_neo/blackrock/FileSpec2.3001.ns5'
    
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
                print(anasig)
            #~ print(len(seg.spiketrains))
            for sptr in seg.spiketrains:
                print(sptr)
                #~ pass

    
if __name__ == '__main__':
    #~ test_BlackrockRawIO()
    test_BlackrockIO()
