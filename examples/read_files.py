# -*- coding: utf-8 -*-
"""
This is an example for reading files with neo.io
"""

import urllib

import neo


# Plexon files
distantfile = 'https://portal.g-node.org/neo/plexon/File_plexon_3.plx'
localfile = './File_plexon_3.plx'
urllib.urlretrieve(distantfile, localfile)

#create a reader
reader = neo.io.PlexonIO(filename='File_plexon_3.plx')
# read the blocks
blks = reader.read(cascade=True, lazy=False)
print blks
# acces to segments
for blk in blks:
    for seg in blk.segments:
        print seg
        for asig in seg.analogsignals:
            print asig
        for st in seg.spiketrains:
            print st


# CED Spike2 files
distantfile = 'https://portal.g-node.org/neo/spike2/File_spike2_1.smr'
localfile = './File_spike2_1.smr'
urllib.urlretrieve(distantfile, localfile)

#create a reader
reader = neo.io.Spike2IO(filename='File_spike2_1.smr')
# read the block
bl = reader.read(cascade=True, lazy=False)[0]
print bl
# acces to segments
for seg in bl.segments:
    print seg
    for asig in seg.analogsignals:
        print asig
    for st in seg.spiketrains:
        print st
