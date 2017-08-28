# -*- coding: utf-8 -*-
"""
This is an example for reading files with neo.io
"""

import urllib

from neo.rawio import PlexonRawIO, Spike2RawIO



# Plexon files
distantfile = 'https://portal.g-node.org/neo/plexon/File_plexon_3.plx'
localfile = './File_plexon_3.plx'
urllib.request.urlretrieve(distantfile, localfile)

# create a reader
reader = PlexonRawIO(filename='File_plexon_3.plx')
reader.parse_header()
print(reader)
print(reader.header)

for k, v in reader.header.items():
    print(k, v)

#Read signal chunks
channel_indexes = None #could be channel_indexes = [0]
raw_sigs = reader.get_analogsignal_chunk(block_index=0, seg_index=0, 
                        i_start=1024, i_stop=2048, channel_indexes=channel_indexes)
float_sigs = reader.rescale_signal_raw_to_float(raw_sigs, dtype='float64')
sampling_rate = reader.get_signal_sampling_rate()
t_start = reader.get_signal_t_start(block_index=0, seg_index=0)
units =reader.header['signal_channels'][0]['units']
print(raw_sigs.shape, raw_sigs.dtype)
print(float_sigs.shape, float_sigs.dtype)
print(sampling_rate, t_start, units)



# read the blocks
#~ blks = reader.read(cascade=True, lazy=False)
#~ print (blks)
#~ # access to segments
#~ for blk in blks:
    #~ for seg in blk.segments:
        #~ print (seg)
        #~ for asig in seg.analogsignals:
            #~ print (asig)
        #~ for st in seg.spiketrains:
            #~ print (st)


#~ # CED Spike2 files
#~ distantfile = 'https://portal.g-node.org/neo/spike2/File_spike2_1.smr'
#~ localfile = './File_spike2_1.smr'
#~ urllib.request.urlretrieve(distantfile, localfile)

#~ # create a reader
#~ reader = neo.io.Spike2IO(filename='File_spike2_1.smr')
#~ # read the block
#~ bl = reader.read(cascade=True, lazy=False)[0]
#~ print (bl)
#~ # access to segments
#~ for seg in bl.segments:
    #~ print (seg)
    #~ for asig in seg.analogsignals:
        #~ print (asig)
    #~ for st in seg.spiketrains:
        #~ print (st)
