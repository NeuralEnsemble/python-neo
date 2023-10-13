"""
Creating simple plots from various Neo structures
=================================================

It includes a function that generates toy data.
"""

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

    # Create synthetic data
    for seg in block.segments:
        feature_pos = np.random.randint(0, data_samples - feature_samples)

        # Analog signals: Noise with a single sinewave feature
        
        sig = np.random.rand(data_samples, n_channels)
        for channel_idx in range(n_channels):
            wave = np.random.randint(-5,6) * np.sin(np.linspace(0, 2 * np.pi, feature_samples))
            sig[feature_pos:feature_pos + feature_samples, channel_idx] += wave

        signal = neo.AnalogSignal(sig * pq.mV, sampling_rate=1 * pq.kHz)
        seg.analogsignals.append(signal)

        feature_time = feature_pos / data_samples
        spiketrains = []
        for unit in range(n_units):
            random_spikes = np.random.rand(20)
            feature_spikes = np.random.rand(5) * feature_len + feature_time
            spikes = np.hstack([random_spikes, feature_spikes])
            train = neo.SpikeTrain(spikes * pq.s, 1 * pq.s)
            spiketrains.append(train)

        seg.spiketrains.extend(spiketrains)

        units = []
        for i, spiketrain in enumerate(spiketrains):
            unit = neo.Group([spiketrain], name=f"Neuron #{i + 1}")
            units.append(unit)

        for unit in units:
            unit.add(neo.ChannelView(signal, index=list(range(n_channels)), name='Channel Group'))
        
        block.groups.extend(units)

    return block


block = generate_block()

# In this example, we treat each segment in turn, averaging over the channels
# in each:
for seg in block.segments:
    print("Analysing segment %d" % seg.index)

    siglist = seg.analogsignals
    time_points = siglist[0].times
    avg = np.mean(siglist[0], axis=1)  # Average over signals of Segment

    plt.figure()
    plt.plot(time_points, avg)
    plt.title("Peak response in segment %d: %f" % (seg.index, avg.max()))

# The second alternative is spatial traversal of the data (by channel), with
n_channels = np.shape(block.segments[0].analogsignals[0])[1]
siglist=[]
for seg in block.segments:
    siglist.append(seg.analogsignals[0])
    time_points = seg.analogsignals[0].times

final_siglist = np.concatenate((siglist), axis=1)

for channel in range(n_channels):
    print(f'Analyzing channel {channel}')
    avg = np.mean(final_siglist[:,channel::n_channels],axis=1)
    plt.figure()
    plt.plot(time_points, avg)
    plt.title(f"Average response on channel {channel}")


# There are three ways to access the spike train data: by Segment

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


