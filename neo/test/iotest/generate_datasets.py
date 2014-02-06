# -*- coding: utf-8 -*-
'''
Generate datasets for testing
'''

# needed for python 3 compatibility
from __future__ import absolute_import

from datetime import datetime
import logging

import numpy as np
from numpy.random import rand
import quantities as pq

from neo.core import (AnalogSignal, Block, EpochArray, EventArray,
                      RecordingChannel, Segment, SpikeTrain,
                      class_by_name)
from neo.io.tools import populate_RecordingChannel, iteritems
from neo.description import (classes_necessary_attributes,
                             classes_recommended_attributes)


TEST_ANNOTATIONS = [1, 0, 1.5, "this is a test", datetime.now(), None]


def generate_one_simple_block(block_name='block_0', nb_segment=3,
                              supported_objects=[], **kws):
    if supported_objects and Block not in supported_objects:
        raise ValueError('Block must be in supported_objects')
    bl = Block()  # name = block_name)

    objects = supported_objects
    if Segment in objects:
        for s in range(nb_segment):
            seg = generate_one_simple_segment(seg_name="seg" + str(s),
                                              supported_objects=objects, **kws)
            bl.segments.append(seg)

    if RecordingChannel in objects:
        populate_RecordingChannel(bl)

    bl.create_many_to_one_relationship()
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
    if supported_objects and Segment not in supported_objects:
        raise ValueError('Segment must be in supported_objects')
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

    seg.create_many_to_one_relationship()
    return seg


def generate_from_supported_objects(supported_objects):
    #~ create_many_to_one_relationship
    if not supported_objects:
        raise ValueError('No objects specified')
    objects = supported_objects
    if Block in supported_objects:
        higher = generate_one_simple_block(supported_objects=objects)

        # Chris we do not create RC and RCG if it is not in objects
        # there is a test in generate_one_simple_block so I removed
        #finalize_block(higher)

    elif Segment in objects:
        higher = generate_one_simple_segment(supported_objects=objects)
    else:
        #TODO
        return None

    higher.create_many_to_one_relationship()
    return higher


def get_fake_value(name, datatype, dim=0, dtype='float', seed=None):
    """
    Returns default value for a given attribute based on description.py

    If seed is not None, use the seed to set the random number generator.
    """
    if seed is not None:
        np.random.seed(seed)

    if datatype == str:
        return str(np.random.randint(100000))
    if datatype == int:
        return np.random.randint(100)
    if datatype == float:
        return 1000. * np.random.random()
    if datatype == datetime:
        return datetime.now()

    if name == 't_start':
        if datatype != pq.Quantity or dim:
            raise ValueError('t_start must be a 0D Quantity, ' +
                             'not a %sD %s' % (dim, datatype))
        return 0.0 * pq.millisecond

    if name == 't_stop':
        if datatype != pq.Quantity or dim:
            raise ValueError('t_stop must be a 0D Quantity, ' +
                             'not a %sD %s' % (dim, datatype))
        return 1.0 * pq.millisecond

    if name == 'sampling_rate':
        if datatype != pq.Quantity or dim:
            raise ValueError('sampling_rate must be a 0D Quantity, ' +
                             'not a %sD %s' % (dim, datatype))
        return 10000.0 * pq.Hz

    # only put array types below here
    size = []
    for i in range(int(dim)):
        size.append(np.random.randint(100) + 1)
    arr = np.random.random(size)

    if datatype == pq.Quantity:
        return arr * pq.millisecond  # let it be ms
    if datatype == np.ndarray:
        return np.array(arr, dtype=dtype)

    # we have gone through everything we know, so it must be something invalid
    raise ValueError('Unknown name/datatype combination %s %s' % (name,
                                                                  datatype))


def fake_neo(obj_type="Block", cascade=True, _follow_links=True):
    """ Create a fake NEO object of a given type. Follows one-to-many
    and many-to-many relationships if cascade. RC, when requested cascade, will
    not create RGCs to avoid dead-locks.

    _follow_links - an internal variable, indicates whether to create objects
    with 'implicit' relationships, to avoid duplications. Do not use it. """
    kwargs = {}  # assign attributes

    attrs = (classes_necessary_attributes[obj_type] +
             classes_recommended_attributes[obj_type])
    for attr in attrs:
        kwargs[attr[0]] = get_fake_value(*attr)
    if 'times' in kwargs and 'signal' in kwargs:
        kwargs['times'] = kwargs['times'][:len(kwargs['signal'])]
        kwargs['signal'] = kwargs['signal'][:len(kwargs['times'])]

    obj = class_by_name[obj_type](**kwargs)

    if cascade:
        if obj_type == "Block":
            _follow_links = False
        for childname in getattr(obj, '_child_objects', []):
            child = fake_neo(childname, cascade, _follow_links)
            if not _follow_links and obj_type in child._parent_objects[1:]:
                continue
            setattr(obj, childname.lower()+"s", [child])

    # need to manually create 'implicit' connections
    if obj_type == "Block" and cascade:
        # connect a unit to the spike and spike train
        u = obj.recordingchannelgroups[0].units[0]
        st = obj.segments[0].spiketrains[0]
        sp = obj.segments[0].spikes[0]
        u.spiketrains.append(st)
        u.spikes.append(sp)
        # connect RCG with ASA
        asa = obj.segments[0].analogsignalarrays[0]
        obj.recordingchannelgroups[0].analogsignalarrays.append(asa)
        # connect RC to AS, IrSAS and back to RGC
        rc = obj.recordingchannelgroups[0].recordingchannels[0]
        rc.recordingchannelgroups.append(obj.recordingchannelgroups[0])
        rc.analogsignals.append(obj.segments[0].analogsignals[0])
        seg = obj.segments[0]
        rc.irregularlysampledsignals.append(seg.irregularlysampledsignals[0])

    # add some annotations, 80%
    at = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
               range(len(TEST_ANNOTATIONS))])
    obj.annotate(**at)

    obj.create_many_to_one_relationship()

    return obj
