# -*- coding: utf-8 -*-
'''
Generate datasets for testing
'''

# needed for python 3 compatibility
from __future__ import absolute_import

import numpy as np
from numpy.random import rand
import quantities as pq

from neo.core import (AnalogSignal, Block, EpochArray, EventArray,
                      RecordingChannel, Segment, SpikeTrain)
from neo.io.tools import (create_many_to_one_relationship,
                          populate_RecordingChannel, iteritems)


def generate_one_simple_block(block_name='block_0', nb_segment=3,
                              supported_objects=[], **kws):
    bl = Block()  # name = block_name)

    objects = supported_objects
    if Segment in objects:
        for s in range(nb_segment):
            seg = generate_one_simple_segment(seg_name="seg" + str(s),
                                              supported_objects=objects, **kws)
            bl.segments.append(seg)

    if RecordingChannel in objects:
        populate_RecordingChannel(bl)

    return bl


def generate_one_simple_segment(seg_name='segment 0',
                                supported_objects=[],
                                nb_analogsignal=4,
                                t_start=0.*pq.s,
                                sampling_rate=10*pq.kHz,
                                duration=6.*pq.s,

                                nb_spiketrain=6,
                                spikerate_range=[.5*pq.Hz, 12*pq.Hz],

                                event_array_types={'stim': ['a', 'b',
                                                            'c', 'd'],
                                                   'enter_zone': ['one',
                                                                  'two'],
                                                   'color': ['black',
                                                             'yellow',
                                                             'green'],
                                                   },
                                event_array_size_range=[5, 20],

                                epoch_array_types={'animal state': ['Sleep',
                                                                    'Freeze',
                                                                    'Escape'],
                                                   'light': ['dark',
                                                             'lighted']
                                                   },
                                epoch_array_duration_range=[.5, 3.],

                                ):
    seg = Segment(name=seg_name)
    if AnalogSignal in supported_objects:
        for a in range(nb_analogsignal):
            anasig = AnalogSignal(rand(int(sampling_rate * duration)),
                                  sampling_rate=sampling_rate, t_start=t_start,
                                  units=pq.mV, channel_index=a,
                                  name='sig %d for segment %s' % (a, seg.name))
            seg.analogsignals.append(anasig)

    if SpikeTrain in supported_objects:
        for s in range(nb_spiketrain):
            spikerate = rand()*np.diff(spikerate_range)
            spikerate += spikerate_range[0].magnitude
            #spikedata = rand(int((spikerate*duration).simplified))*duration
            #sptr = SpikeTrain(spikedata,
            #                  t_start=t_start, t_stop=t_start+duration)
            #                  #, name = 'spiketrain %d'%s)
            spikes = rand(int((spikerate*duration).simplified))
            spikes.sort()  # spikes are supposed to be an ascending sequence
            sptr = SpikeTrain(spikes*duration,
                              t_start=t_start, t_stop=t_start+duration)
            sptr.annotations['channel_index'] = s
            seg.spiketrains.append(sptr)

    if EventArray in supported_objects:
        for name, labels in iteritems(event_array_types):
            ea_size = rand()*np.diff(event_array_size_range)
            ea_size += event_array_size_range[0]
            labels = np.array(labels, dtype='S')
            labels = labels[(rand(ea_size)*len(labels)).astype('i')]
            ea = EventArray(times=rand(ea_size)*duration, labels=labels)
            seg.eventarrays.append(ea)

    if EpochArray in supported_objects:
        for name, labels in iteritems(epoch_array_types):
            t = 0
            times = []
            durations = []
            while t < duration:
                times.append(t)
                dur = rand()*np.diff(epoch_array_duration_range)
                dur += epoch_array_duration_range[0]
                durations.append(dur)
                t = t+dur
            labels = np.array(labels, dtype='S')
            labels = labels[(rand(len(times))*len(labels)).astype('i')]
            epa = EpochArray(times=pq.Quantity(times, units=pq.s),
                             durations=pq.Quantity([x[0] for x in durations],
                                                   units=pq.s),
                             labels=labels,
                             )
            seg.epocharrays.append(epa)

    # TODO : Spike, Event, Epoch

    return seg


def generate_from_supported_objects(supported_objects):
    #~ create_many_to_one_relationship
    objects = supported_objects
    if Block in objects:
        higher = generate_one_simple_block(supported_objects=objects)

        # Chris we do not create RC and RCG if it is not in objects
        # there is a test in generate_one_simple_block so I removed
        #finalize_block(higher)

    elif Segment in objects:
        higher = generate_one_simple_segment(supported_objects=objects)
    else:
        #TODO
        return None

    create_many_to_one_relationship(higher)
    return higher
