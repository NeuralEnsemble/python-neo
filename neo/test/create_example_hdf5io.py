"""
This script generates an HDF5 file using the NeoHdf5IO, for the purpose
of testing conversion from Neo 0.3/0.4 to 0.5.

"""
from datetime import datetime
import numpy as np
import neo
from quantities import mV, ms, kHz, s, nA

blocks = [neo.Block(name="block1",
                    file_datetime=datetime.now(),
                    index=1234,
                    foo="bar"),
          neo.Block(name="block2",
                    rec_datetime=datetime.now())]

# Block 1: arrays
segments0 = [neo.Segment(name="seg1{0}".format(i))
             for i in range(1, 4)]
blocks[0].segments = segments0
for j, segment in enumerate(segments0):
    segment.block = blocks[0]
    segment.analogsignalarrays = [neo.AnalogSignalArray(
                                        np.random.normal(-60.0 + j + i, 10.0, size=(1000, 4)),
                                        units=mV,
                                        sampling_rate=1*kHz
                                  ) for i in range(2)]
    segment.spiketrains = [neo.SpikeTrain(np.arange(100 + j + i, 900, 10.0), t_stop=1000*ms, units=ms)
                           for i in range(4)]
    # todo: add spike waveforms
    segment.eventarrays = [neo.EventArray(np.arange(0, 30, 10)*s,
                                          labels=np.array(['trig0', 'trig1', 'trig2']))]
    segment.epocharrays = [neo.EpochArray(times=np.arange(0, 30, 10)*s,
                                          durations=[10, 5, 7]*ms,
                                          labels=np.array(['btn0', 'btn1', 'btn2']))]
    segment.irregularlysampledsignals = [neo.IrregularlySampledSignal([0.01, 0.03, 0.12]*s, [4, 5, 6]*nA),
                                         neo.IrregularlySampledSignal([0.01, 0.03, 0.12]*s, [3, 4, 3]*nA),
                                         neo.IrregularlySampledSignal([0.02, 0.05, 0.15]*s, [3, 4, 3]*nA)]

# Block 2: singletons
segments1 = [neo.Segment(name="seg2{0}".format(i))
             for i in range(1, 3)]
blocks[1].segments = segments1
recordingchannels = [neo.RecordingChannel(index=i) for i in range(8)]
units = [neo.Unit(name="unit{0}".format(i)) for i in range(7)]
rcgs = [neo.RecordingChannelGroup(name="electrode1"),
        neo.RecordingChannelGroup(name="electrode2"),
        neo.RecordingChannelGroup(name="my_favourite_channels")]
rcgs[0].recordingchannels = recordingchannels[:4]
rcgs[0].units = units[:2]
rcgs[1].recordingchannels = recordingchannels[4:]
rcgs[1].units = units[2:]
rcgs[2].recordingchannels = [recordingchannels[1], recordingchannels[3]]
blocks[1].recordingchannelgroups = rcgs
for j, segment in enumerate(segments1):
    segment.block = blocks[1]
    segment.analogsignals = [neo.AnalogSignal(
                                  np.random.normal(-60.0 + j + i, 10.0, size=(1000,)),
                                  units=mV,
                                  sampling_rate=1*kHz
                             ) for i in range(8)]
    for rc, signal in zip(recordingchannels, segment.analogsignals):
        rc.analogsignals.append(signal)
    segment.spiketrains = [neo.SpikeTrain(np.arange(100 + j + i, 900, 10.0), t_stop=1000*ms, units=ms)
                           for i in range(7)]
    rcgs[0].units[0].spiketrains.append(segment.spiketrains[0])
    rcgs[0].units[1].spiketrains.append(segment.spiketrains[1])
    for unit, st in zip(rcgs[1].units, segment.spiketrains[2:]):
        unit.spiketrains.append(st)

# todo: add some Event, Epoch objects

io = neo.io.NeoHdf5IO("neo_hdf5_example.h5")
io.write_all_blocks(blocks)
