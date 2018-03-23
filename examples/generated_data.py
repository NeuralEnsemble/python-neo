# -*- coding: utf-8 -*-
"""
This is an example for creating simple plots from various Neo structures.
It includes a function that generates toy data.
"""
from __future__ import division  # Use same division in Python 2 and 3

import numpy as np
import quantities as pq
from matplotlib import pyplot as plt

import neo


def generate_block(n_segments=3, n_channels=4, n_units=3,
                   data_samples=1000, feature_samples=100):
    """
    Generate a block with a single recording channel group and a number of
    segments, recording channels and units with associated analog signals
    and spike trains.
    """
    feature_len = feature_samples / data_samples

    # Create Block to contain all generated data
    block = neo.Block()

    # Create multiple Segments
    block.segments = [neo.Segment(index=i) for i in range(n_segments)]
    # Create multiple ChannelIndexes
    block.channel_indexes = [neo.ChannelIndex(name='C%d' % i, index=i) for i in range(n_channels)]

    # Attach multiple Units to each ChannelIndex
    for channel_idx in block.channel_indexes:
        channel_idx.units = [neo.Unit('U%d' % i) for i in range(n_units)]

    # Create synthetic data
    for seg in block.segments:
        feature_pos = np.random.randint(0, data_samples - feature_samples)

        # Analog signals: Noise with a single sinewave feature
        wave = 3 * np.sin(np.linspace(0, 2 * np.pi, feature_samples))
        for channel_idx in block.channel_indexes:
            sig = np.random.randn(data_samples)
            sig[feature_pos:feature_pos + feature_samples] += wave

            signal = neo.AnalogSignal(sig * pq.mV, sampling_rate=1 * pq.kHz)
            seg.analogsignals.append(signal)
            channel_idx.analogsignals.append(signal)

            # Spike trains: Random spike times with elevated rate in short period
            feature_time = feature_pos / data_samples
            for u in channel_idx.units:
                random_spikes = np.random.rand(20)
                feature_spikes = np.random.rand(5) * feature_len + feature_time
                spikes = np.hstack([random_spikes, feature_spikes])

                train = neo.SpikeTrain(spikes * pq.s, 1 * pq.s)
                seg.spiketrains.append(train)
                u.spiketrains.append(train)

    block.create_many_to_one_relationship()
    return block


block = generate_block()

# In this example, we treat each segment in turn, averaging over the channels
# in each:
for seg in block.segments:
    print("Analysing segment %d" % seg.index)

    siglist = seg.analogsignals
    time_points = siglist[0].times
    avg = np.mean(siglist, axis=0)  # Average over signals of Segment

    plt.figure()
    plt.plot(time_points, avg)
    plt.title("Peak response in segment %d: %f" % (seg.index, avg.max()))

# The second alternative is spatial traversal of the data (by channel), with
# averaging over trials. For example, perhaps you wish to see which physical
# location produces the strongest response, and each stimulus was the same:

# There are multiple ChannelIndex objects connected to the block, each
# corresponding to a a physical electrode
for channel_idx in block.channel_indexes:
    print("Analysing channel %d: %s" % (channel_idx.index, channel_idx.name))

    siglist = channel_idx.analogsignals
    time_points = siglist[0].times
    avg = np.mean(siglist, axis=0)  # Average over signals of RecordingChannel

    plt.figure()
    plt.plot(time_points, avg)
    plt.title("Average response on channel %d" % channel_idx.index)

# There are three ways to access the spike train data: by Segment,
# by ChannelIndex or by Unit.

# By Segment. In this example, each Segment represents data from one trial,
# and we want a peristimulus time histogram (PSTH) for each trial from all
# Units combined:
for seg in block.segments:
    print("Analysing segment %d" % seg.index)
    stlist = [st - st.t_start for st in seg.spiketrains]
    count, bins = np.histogram(np.hstack(stlist))
    plt.figure()
    plt.bar(bins[:-1], count, width=bins[1] - bins[0])
    plt.title("PSTH in segment %d" % seg.index)

# By Unit. Now we can calculate the PSTH averaged over trials for each Unit:
for unit in block.list_units:
    stlist = [st - st.t_start for st in unit.spiketrains]
    count, bins = np.histogram(np.hstack(stlist))
    plt.figure()
    plt.bar(bins[:-1], count, width=bins[1] - bins[0])
    plt.title("PSTH of unit %s" % unit.name)

# By ChannelIndex. Here we calculate a PSTH averaged over trials by
# channel location, blending all Units:
for chx in block.channel_indexes:
    stlist = []
    for unit in chx.units:
        stlist.extend([st - st.t_start for st in unit.spiketrains])
    count, bins = np.histogram(np.hstack(stlist))
    plt.figure()
    plt.bar(bins[:-1], count, width=bins[1] - bins[0])
    plt.title("PSTH blend of recording channel group %s" % chx.name)

plt.show()
