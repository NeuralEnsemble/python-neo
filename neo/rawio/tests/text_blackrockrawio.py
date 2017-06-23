
import neo
#~ import neo.rawio.blackrockrawio
from  neo.rawio.blackrockrawio import BlackrockRawIO
import numpy as np

import time

def test_BlackrockRawIO():
    #~ filename = '/media/sgarcia/SamCNRS/DataSpikeSorting/elodie/Dataset 3/20161124-112218-001'
    filename = '/media/sgarcia/SamCNRS/DataSpikeSorting/elodie/Nouveaux_datasets/micro_VS10_SAB 1 600ms/20160627-155334-001'
    
    reader = BlackrockRawIO(filename=filename)
    reader.parse_header()
    
    assert reader.block_count()==1
    
    assert reader.segment_count(0)==1
    
    # Acces 10000 chunk of 1024 samples for 10 channels
    nb_chunk = 10000
    chunksize = 1024
    channel_indexes = np.arange(20,30)
    
    t0 = time.perf_counter()
    for i in range(nb_chunk):
        i_start = i*chunksize
        i_stop = (i+1)*chunksize
        raw_chunk = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=i_start, i_stop=i_stop,  channel_indexes=channel_indexes)
        #~ print(sig_chunk.shape)
    t1 = time.perf_counter()
    print('acces {} chunk of {} samples in {:0.3f} s'.format(nb_chunk,chunksize, t1-t0))
    
    
    
    channel_names =['b6']
    raw_chunk = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=0, i_stop=1024,  channel_names=channel_names)
    
    float_chunk = reader.rescale_raw_to_float(raw_chunk, dtype='float64', channel_names=channel_names)
    
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(raw_chunk)
    axs[1].plot(float_chunk)
    plt.show()
    
    
    
    
    
if __name__ == '__main__':
    test_BlackrockRawIO()