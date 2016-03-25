# -*- coding: utf-8 -*-
"""
This is an example for reading files with neo.io
"""

import urllib

import neo


# Plexon files
data_path = '/Users/Summit/Dropbox/Coding_Projects_Data/python-neo/tdt/aep_05'

#create a reader
reader = neo.io.TdtIO(dirname=data_path)
# read the blocks
blks = reader.read_block(cascade=True, lazy=False)
print blks
# acces to segments
for blk in blks:
    for seg in blk.segments:
        print seg
        for asig in seg.analogsignals:
            print asig
        for st in seg.spiketrains:
            print st


