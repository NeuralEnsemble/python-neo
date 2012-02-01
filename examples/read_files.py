"""
This is an example for reading files with neo.io
"""

import neo
import urllib




# Plexon files
distantfile = 'https://portal.g-node.org/neo/plexon/File_plexon_3.plx'
localfile = './File_plexon_3.plx'
urllib.urlretrieve(distantfile, localfile)

#create a reader
reader = neo.io.PlexonIO(filename = 'File_plexon_3.plx')
# read the block
bl = reader.read(cascade = True, lazy = False)
print bl
# acces to segments
for seg in bl.segments:
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
reader = neo.io.Spike2IO(filename = 'File_spike2_1.smr')
# read the block
bl = reader.read(cascade = True, lazy = False)
print bl
# acces to segments
for seg in bl.segments:
    print seg
    for asig in seg.analogsignals:
        print asig
    for st in seg.spiketrains:
        print st

