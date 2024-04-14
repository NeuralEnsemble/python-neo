"""
Demonstration of lazy load and proxy objects
============================================

"""

################################################
# Import our packages first
# It is often nice to have units so we will also
# import quantities

import urllib
import neo
import quantities as pq
import numpy as np


###############################################
# Let's get a file
# NeuralEnsemble maintains a wide variety of small test
# datasets that are free to use. We can use urllib to pull
# down one of these files for use

url_repo = "https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data/raw/master/"
# Get med file
distantfile = url_repo + "micromed/File_micromed_1.TRC"
localfile = "./File_micromed_1.TRC"
urllib.request.urlretrieve(distantfile, localfile)

##################################################
# create a reader
# creating a reader for neo is easy it just requires using
# the name of the desired reader and providing either a filename
# or a directory name (reader dependent). Since we got a micromed
# file we will use MicromedIO.

reader = neo.MicromedIO(filename="File_micromed_1.TRC")
reader.parse_header()

############################################################
# as always we can look view some interesting information about the
# metadata and structure of a file just by printing the reader and
# it's header
print(reader)
print(f"Header information: {reader.header}")


#####################################################
# Now let's make a function that we want to apply to
# look at lazy vs eager uses of the API


def apply_my_fancy_average(sig_list):
    """basic average along triggers and then channels
    here we go back to numpy with magnitude
    to be able to use np.stack.

    Because neo uses quantities to keep track of units
    we can always get just the magnitude of an array
    with `.magnitude`
    """
    sig_list = [s.magnitude for s in sig_list]
    sigs = np.stack(sig_list, axis=0)
    return np.mean(np.mean(sigs, axis=0), axis=1)


#################################################
# Let's set our limits for both cases. We will
# use quantities to include time dimensions.

lim_start = -20 * pq.ms  # 20 milliseconds before
lim_end = +20 * pq.ms  # 20 milliseconds after

##################################################
# We start with eager (where `lazy=False`.) Everything
# is loaded into memory. We will read a segment of data.
# This includes analog signal data and events data
# (final contents of a segment are dependent on the
# underlying IO being used)


seg = reader.read_segment(lazy=False)
triggers = seg.events[0]
anasig = seg.analogsignals[0]  # here anasig contain the whole recording in memory
all_sig_chunks = []
for t in triggers.times:
    t0, t1 = (t + lim_start), (t + lim_end)
    anasig_chunk = anasig.time_slice(t0, t1)
    all_sig_chunks.append(anasig_chunk)

# After pulling all data into memory and then iterating through triggers
# we end by doing our average
m1 = apply_my_fancy_average(all_sig_chunks)

#####################################################
# Here we do `lazy=True`, i.e. we do lazy loading. We
# only load the data that we want into memory
# and we use a proxy object for our analogsignal until we
# load it chunk by chunk (no running out of memory!)

seg = reader.read_segment(lazy=True)
triggers = seg.events[0].load(time_slice=None)  # this load all triggers in memory
anasigproxy = seg.analogsignals[0]  # this is a proxy
all_sig_chunks = []
for t in triggers.times:
    t0, t1 = (t + lim_start), (t + lim_end)
    # at this step we load actual data into memory, but notice that we only load one
    # chunk of data at a time, so we reduce the memory strain
    anasig_chunk = anasigproxy.load(time_slice=(t0, t1))  # here real data are loaded
    all_sig_chunks.append(anasig_chunk)

# Finally we apply the same average as we did above
m2 = apply_my_fancy_average(all_sig_chunks)

##########################################################
# We see that either way the result is the same, but
# we do not exhaust our RAM/memory
print(f"Eagerly loading data and averaging: {m1}")
print(f"Lazy loading data and average {m2}")
