"""
Reading files with neo.rawio
============================

compare with read_files_neo_io.py
"""

###########################################################
# First we import a RawIO from neo.rawio
# For this example we will use PlexonRawIO

import urllib
from neo.rawio import PlexonRawIO

##############################################################
# Get Plexon files

# We will be pulling these files down, but if you have a local file
# then all you need to do is specify the file location on your
# computer. NeuralEnsemble keeps a wide variety of freely accesible, small
# test files that can be used. So for this example we will take advantage of that
# fact.

url_repo = "https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data/raw/master/"
distantfile = url_repo + "plexon/File_plexon_3.plx"
localfile = "File_plexon_3.plx"
urllib.request.urlretrieve(distantfile, localfile)

###############################################################
# Create a reader

# All it takes to create a reader is giving the filename (or dirname)
# Then we need to do the slow step of parsing the header with the
# `parse_header` function. This collects metadata as well as
# make all future steps much faster for us

reader = PlexonRawIO(filename="File_plexon_3.plx")
reader.parse_header()
print(reader)
# we can view metadata in the header
print(reader.header)

###############################################################
# Read signal chunks
# This is how we read raw data. We choose indices that we want or
# we can use None to mean look at all channels. We also need to
# specify the block of data (block_index) as well as the segment
# (seg_index). Then we give the index start and stop. Since we
# often think in time: to go from time to index would just require
# the sample rate (so index = time / sampling_rate)

channel_indexes = None  # could be channel_indexes = [0]
raw_sigs = reader.get_analogsignal_chunk(
    block_index=0, seg_index=0, i_start=1024, i_stop=2048, channel_indexes=channel_indexes
)

# raw_sigs are not voltages so to convert to voltages we do the follwing
float_sigs = reader.rescale_signal_raw_to_float(raw_sigs, dtype="float64")

# We can see that the shapes are the same, but that the datatypes
# are different once we've rescaled our data
print("Raw Data: ", raw_sigs.shape, raw_sigs.dtype)
print("Scaled Data: ", float_sigs.shape, float_sigs.dtype)

###############################################################
# Each rawio gives you access to lots of information about your data
# some of this information comes from functions
# other information is stored as metadata in the reader.header

sampling_rate = reader.get_signal_sampling_rate()
# Like above we need to indicate the block and segment
t_start = reader.get_signal_t_start(block_index=0, seg_index=0)
units = reader.header["signal_channels"][0]["units"]

# and we can display all of this information
print(f"{sampling_rate=}, {t_start=}, {units=}")


####################################################################
# Some rawio's and file formats also provide information about spikes
# If a rawio can't read this data it will raise an error, but lucky
# for us PlexonRawIO does have spikes data!!

# Count units and spikes per unit
nb_unit = reader.spike_channels_count()
print(f"nb_unit: {nb_unit}\n")  # nb_unit stands for number of units
print("spike_channel_index     nb_spike")
for spike_channel_index in range(nb_unit):
    nb_spike = reader.spike_count(block_index=0, seg_index=0, spike_channel_index=spike_channel_index)
    print(f"{spike_channel_index}: {nb_spike}\n")

# Read spike times and rescale (just like analogsignal above)
spike_timestamps = reader.get_spike_timestamps(
    block_index=0, seg_index=0, spike_channel_index=0, t_start=0.0, t_stop=10.0
)

print(f"{spike_timestamps.shape=}\n{spike_timestamps.dtype=}\n{spike_timestamps[:5]=}\n")
spike_times = reader.rescale_spike_timestamp(spike_timestamps, dtype="float64")
print(f"{spike_times.shape=}\n{spike_times.dtype=}\n{spike_times[:5]}\n")

#######################################################################
# Some file formats can also give waveform information. We are lucky
# again our file has waveform data!! We forms are a 3d dataset of
# (nb_spike, nb_channel, nb_sample)

# Read spike waveforms
raw_waveforms = reader.get_spike_raw_waveforms(
    block_index=0, seg_index=0, spike_channel_index=0, t_start=0.0, t_stop=10.0
)
print(f"{raw_waveforms.shape=}\n{raw_waveforms.dtype=}\n{raw_waveforms[0, 0, :4]=}\n")
float_waveforms = reader.rescale_waveforms_to_float(raw_waveforms, dtype="float32", spike_channel_index=0)
print(f"{float_waveforms.shape=}\n{float_waveforms.dtype=}{float_waveforms[0, 0, :4]=}\n")

#########################################################################
# RawIOs can also read event timestamps. But looks like our luck ran out
# let's grab a new file to see this feature

# Read event timestamps and times (take another file)
distantfile = url_repo + "plexon/File_plexon_2.plx"
localfile = "File_plexon_2.plx"
urllib.request.urlretrieve(distantfile, localfile)

#########################################################################
# Since this is a new file we need to read initialize our reader as well
# as parse the header

reader = PlexonRawIO(filename="File_plexon_2.plx")
reader.parse_header()
# if we look at this header we see it is different than the header above
print(reader.header)

###########################################################################
# Now let's look at some event data. This could be things like stimuli applied
# during the course of an ephys recording

nb_event_channel = reader.event_channels_count()
print(f"nb_event_channel: {nb_event_channel}")
# now iterate through the channels
for chan_index in range(nb_event_channel):
    nb_event = reader.event_count(block_index=0, seg_index=0, event_channel_index=chan_index)
    print(f"chan_index: {chan_index} nb_event: {nb_event}\n")


###############################################################################
# Finally we can get our actual event timestamps. Some file formats provide the
# real timestamps (timestamps in s/ms) others have raw timestamps (in samples)
# so we can do the same style of functions as above. Get the raw timestamps
# and convert to real times with a rescale function.

ev_timestamps, ev_durations, ev_labels = reader.get_event_timestamps(
    block_index=0, seg_index=0, event_channel_index=0, t_start=None, t_stop=None
)
print(f"{ev_timestamps=}\n{ev_durations=}\n{ev_labels=}\n")
ev_times = reader.rescale_event_timestamp(ev_timestamps, dtype="float64")
print(f"{ev_times=}\n")
