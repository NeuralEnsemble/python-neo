"""
Reading files with neo.io
=========================

"""

####################################################
# Start with package import and getting a datafile

import urllib

import neo

url_repo = "https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data/raw/master/"

# Plexon files
distantfile = url_repo + "plexon/File_plexon_3.plx"
localfile = "File_plexon_3.plx"
urllib.request.urlretrieve(distantfile, localfile)


###################################################
# Now we can create our reader and read some data

# create a reader
reader = neo.io.PlexonIO(filename="File_plexon_3.plx")
# read the blocks
blks = reader.read(lazy=False)
print(blks)

######################################################
# Once we have our blocks we can iterate through each
# block of data and see the contents of all parts of
# that data

# access to segments
for blk in blks:
    for seg in blk.segments:
        print(seg)
        for asig in seg.analogsignals:
            print(asig)
        for st in seg.spiketrains:
            print(st)

#######################################################
# Let's look at another file type

# CED Spike2 files
distantfile = url_repo + "spike2/File_spike2_1.smr"
localfile = "./File_spike2_1.smr"
urllib.request.urlretrieve(distantfile, localfile)
# create a reader
reader = neo.io.Spike2IO(filename="File_spike2_1.smr")

#########################################################
# Despite being a different raw file format we can access
# the data in the same way

# read the block
bl = reader.read(lazy=False)[0]
print(bl)

##########################################################
# Similarly we can view the different types of data within
# the block (AnalogSignals and SpikeTrains)

# access to segments
for seg in bl.segments:
    print(seg)
    for asig in seg.analogsignals:
        print(asig)
    for st in seg.spiketrains:
        print(st)
