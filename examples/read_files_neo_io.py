# -*- coding: utf-8 -*-
"""
This is an example for reading files with neo.io
"""

import urllib

import neo

url_repo = 'https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data/raw/master/'

# Plexon files
distantfile = url_repo + 'plexon/File_plexon_3.plx'
localfile = './File_plexon_3.plx'
urllib.request.urlretrieve(distantfile, localfile)

# create a reader
reader = neo.io.PlexonIO(filename='File_plexon_3.plx')
# read the blocks
blks = reader.read(lazy=False)
print(blks)
# access to segments
for blk in blks:
    for seg in blk.segments:
        print(seg)
        for asig in seg.analogsignals:
            print(asig)
        for st in seg.spiketrains:
            print(st)

# CED Spike2 files
distantfile = url_repo + 'spike2/File_spike2_1.smr'
localfile = './File_spike2_1.smr'
urllib.request.urlretrieve(distantfile, localfile)

# create a reader
reader = neo.io.Spike2IO(filename='File_spike2_1.smr')
# read the block
bl = reader.read(lazy=False)[0]
print(bl)
# access to segments
for seg in bl.segments:
    print(seg)
    for asig in seg.analogsignals:
        print(asig)
    for st in seg.spiketrains:
        print(st)
