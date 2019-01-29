# -*- coding: utf-8 -*-
'''
Generate datasets for testing
'''

# needed for python 3 compatibility
from __future__ import absolute_import

from datetime import datetime

import numpy as np
from numpy.random import rand
import quantities as pq

from neo.core import (AnalogSignal, Block, Epoch, Event, IrregularlySampledSignal, ChannelIndex,
                      Segment, SpikeTrain, Unit, class_by_name)

from neo.core.baseneo import _container_name
from neo.core.dataobject import DataObject

TEST_ANNOTATIONS = [1, 0, 1.5, "this is a test", datetime.fromtimestamp(424242424), None]


def generate_one_simple_block(block_name='block_0', nb_segment=3, supported_objects=[], **kws):
    if supported_objects and Block not in supported_objects:
        raise ValueError('Block must be in supported_objects')
    bl = Block()  # name = block_name)

    objects = supported_objects
    if Segment in objects:
        for s in range(nb_segment):
            seg = generate_one_simple_segment(seg_name="seg" + str(s), supported_objects=objects,
                                              **kws)
            bl.segments.append(seg)

    # if RecordingChannel in objects:
    #    populate_RecordingChannel(bl)

    bl.create_many_to_one_relationship()
    return bl


def generate_one_simple_segment(seg_name='segment 0', supported_objects=[], nb_analogsignal=4,
                                t_start=0. * pq.s, sampling_rate=10 * pq.kHz, duration=6. * pq.s,

                                nb_spiketrain=6, spikerate_range=[.5 * pq.Hz, 12 * pq.Hz],

                                event_types={'stim': ['a', 'b', 'c', 'd'],
                                             'enter_zone': ['one', 'two'],
                                             'color': ['black', 'yellow', 'green'], },
                                event_size_range=[5, 20],

                                epoch_types={'animal state': ['Sleep', 'Freeze', 'Escape'],
                                             'light': ['dark', 'lighted']},
                                epoch_duration_range=[.5, 3.],
                                # this should be multiplied by pq.s, no?

                                array_annotations={'valid': np.array([True, False]),
                                                   'number': np.array(range(5))}

                                ):
    if supported_objects and Segment not in supported_objects:
        raise ValueError('Segment must be in supported_objects')
    seg = Segment(name=seg_name)
    if AnalogSignal in supported_objects:
        for a in range(nb_analogsignal):
            anasig = AnalogSignal(rand(int(sampling_rate * duration)), sampling_rate=sampling_rate,
                                  t_start=t_start, units=pq.mV, channel_index=a,
                                  name='sig %d for segment %s' % (a, seg.name))
            seg.analogsignals.append(anasig)

    if SpikeTrain in supported_objects:
        for s in range(nb_spiketrain):
            spikerate = rand() * np.diff(spikerate_range)
            spikerate += spikerate_range[0].magnitude
            # spikedata = rand(int((spikerate*duration).simplified))*duration
            # sptr = SpikeTrain(spikedata,
            #                  t_start=t_start, t_stop=t_start+duration)
            #                  #, name = 'spiketrain %d'%s)
            spikes = rand(int((spikerate * duration).simplified))
            spikes.sort()  # spikes are supposed to be an ascending sequence
            sptr = SpikeTrain(spikes * duration, t_start=t_start, t_stop=t_start + duration)
            sptr.annotations['channel_index'] = s
            # Randomly generate array_annotations from given options
            arr_ann = {key: value[(rand(len(spikes)) * len(value)).astype('i')] for (key, value) in
                       array_annotations.items()}
            sptr.array_annotate(**arr_ann)
            seg.spiketrains.append(sptr)

    if Event in supported_objects:
        for name, labels in event_types.items():
            evt_size = rand() * np.diff(event_size_range)
            evt_size += event_size_range[0]
            evt_size = int(evt_size)
            labels = np.array(labels, dtype='S')
            labels = labels[(rand(evt_size) * len(labels)).astype('i')]
            evt = Event(times=rand(evt_size) * duration, labels=labels)
            # Randomly generate array_annotations from given options
            arr_ann = {key: value[(rand(evt_size) * len(value)).astype('i')] for (key, value) in
                       array_annotations.items()}
            evt.array_annotate(**arr_ann)
            seg.events.append(evt)

    if Epoch in supported_objects:
        for name, labels in epoch_types.items():
            t = 0
            times = []
            durations = []
            while t < duration:
                times.append(t)
                dur = rand() * (epoch_duration_range[1] - epoch_duration_range[0])
                dur += epoch_duration_range[0]
                durations.append(dur)
                t = t + dur
            labels = np.array(labels, dtype='S')
            labels = labels[(rand(len(times)) * len(labels)).astype('i')]
            assert len(times) == len(durations)
            assert len(times) == len(labels)
            epc = Epoch(times=pq.Quantity(times, units=pq.s),
                        durations=pq.Quantity(durations, units=pq.s),
                        labels=labels,)
            assert epc.times.dtype == 'float'
            # Randomly generate array_annotations from given options
            arr_ann = {key: value[(rand(len(times)) * len(value)).astype('i')] for (key, value) in
                       array_annotations.items()}
            epc.array_annotate(**arr_ann)
            seg.epochs.append(epc)

    # TODO : Spike, Event

    seg.create_many_to_one_relationship()
    return seg


def generate_from_supported_objects(supported_objects):
    # ~ create_many_to_one_relationship
    if not supported_objects:
        raise ValueError('No objects specified')
    objects = supported_objects
    if Block in supported_objects:
        higher = generate_one_simple_block(supported_objects=objects)

        # Chris we do not create RC and RCG if it is not in objects  # there is a test in
        # generate_one_simple_block so I removed  # finalize_block(higher)

    elif Segment in objects:
        higher = generate_one_simple_segment(supported_objects=objects)
    else:
        # TODO
        return None

    higher.create_many_to_one_relationship()
    return higher


def get_fake_value(name, datatype, dim=0, dtype='float', seed=None, units=None, obj=None, n=None,
                   shape=None):
    """
    Returns default value for a given attribute based on neo.core

    If seed is not None, use the seed to set the random number generator.
    """
    if not obj:
        obj = 'TestObject'
    elif not hasattr(obj, 'lower'):
        obj = obj.__name__

    if (name in ['name', 'file_origin', 'description'] and (datatype != str or dim)):
        raise ValueError('%s must be str, not a %sD %s' % (name, dim, datatype))

    if name == 'file_origin':
        return 'test_file.txt'
    if name == 'name':
        return '%s%s' % (obj, get_fake_value('', datatype, seed=seed))
    if name == 'description':
        return 'test %s %s' % (obj, get_fake_value('', datatype, seed=seed))

    if seed is not None:
        np.random.seed(seed)

    if datatype == str:
        return str(np.random.randint(100000))
    if datatype == int:
        return np.random.randint(100)
    if datatype == float:
        return 1000. * np.random.random()
    if datatype == datetime:
        return datetime.fromtimestamp(1000000000 * np.random.random())

    if (name in ['t_start', 't_stop', 'sampling_rate'] and (datatype != pq.Quantity or dim)):
        raise ValueError('%s must be a 0D Quantity, not a %sD %s' % (name, dim, datatype))

    # only put array types below here

    if units is not None:
        pass
    elif name in ['t_start', 't_stop', 'time', 'times', 'duration', 'durations']:
        units = pq.millisecond
    elif name == 'sampling_rate':
        units = pq.Hz
    elif datatype == pq.Quantity:
        units = np.random.choice(['nA', 'mA', 'A', 'mV', 'V'])
        units = getattr(pq, units)

    if name == 'sampling_rate':
        data = np.array(10000.0)
    elif name == 't_start':
        data = np.array(0.0)
    elif name == 't_stop':
        data = np.array(1.0)
    elif n and name == 'channel_indexes':
        data = np.arange(n)
    elif n and name == 'channel_names':
        data = np.array(["ch%d" % i for i in range(n)])
    elif n and obj == 'AnalogSignal':
        if name == 'signal':
            size = []
            for _ in range(int(dim)):
                size.append(np.random.randint(5) + 1)
            size[1] = n
            data = np.random.random(size) * 1000.
    else:
        size = []
        for _ in range(int(dim)):
            if shape is None:
                # To ensure consistency, times, labels and durations need to have the same size
                if name in ["times", "labels", "durations"]:
                    size.append(5)
                else:
                    size.append(np.random.randint(5) + 1)
            else:
                size.append(shape)

        data = np.random.random(size)
        if name not in ['time', 'times']:
            data *= 1000.
    if np.dtype(dtype) != np.float64:
        data = data.astype(dtype)

    if datatype == np.ndarray:
        return data
    if datatype == list:
        return data.tolist()
    if datatype == pq.Quantity:
        return data * units  # set the units

    # Array annotations need to be a dict containing arrays
    if name == 'array_annotations' and datatype == dict:
        # Make sure that array annotations have the correct length
        if obj in ['AnalogSignal', 'IrregularlySampledSignal']:
            length = n if n is not None else 1
        elif obj in ['IrregularlySampledSignal', 'SpikeTrain', 'Epoch', 'Event']:
            length = n
        else:
            raise ValueError("This object cannot have array annotations")
        # Generate array annotations
        valid = np.array([True, False])
        number = np.arange(5)
        arr_ann = {'valid': valid[(rand(length) * len(valid)).astype('i')],
                   'number': number[(rand(length) * len(number)).astype('i')]}

        return arr_ann

    # we have gone through everything we know, so it must be something invalid
    raise ValueError('Unknown name/datatype combination %s %s' % (name, datatype))


def get_fake_values(cls, annotate=True, seed=None, n=None):
    """
    Returns a dict containing the default values for all attribute for
    a class from neo.core.

    If seed is not None, use the seed to set the random number generator.
    The seed is incremented by 1 for each successive object.

    If annotate is True (default), also add annotations to the values.
    """

    if hasattr(cls,
               'lower'):  # is this a test that cls is a string? better to use isinstance(cls,
        # basestring), no?
        cls = class_by_name[cls]
    # iseed is needed below for generation of array annotations
    iseed = None
    kwargs = {}  # assign attributes
    for i, attr in enumerate(cls._necessary_attrs + cls._recommended_attrs):
        if seed is not None:
            iseed = seed + i
        kwargs[attr[0]] = get_fake_value(*attr, seed=iseed, obj=cls, n=n)

    if 'waveforms' in kwargs:  # everything here is to force the kwargs to have len(time) ==
        # kwargs["waveforms"].shape[0]
        if len(kwargs["times"]) != kwargs["waveforms"].shape[0]:
            if len(kwargs["times"]) < kwargs["waveforms"].shape[0]:

                dif = kwargs["waveforms"].shape[0] - len(kwargs["times"])

                new_times = []
                for i in kwargs["times"].magnitude:
                    new_times.append(i)

                np.random.seed(0)
                new_times = np.concatenate([new_times, np.random.random(dif)])
                kwargs["times"] = pq.Quantity(new_times, units=pq.ms)
            else:
                kwargs['times'] = kwargs['times'][:kwargs["waveforms"].shape[0]]

    # IrregularlySampledSignal
    if 'times' in kwargs and 'signal' in kwargs:
        kwargs['times'] = kwargs['times'][:len(kwargs['signal'])]
        kwargs['signal'] = kwargs['signal'][:len(kwargs['times'])]

    if annotate:
        kwargs.update(get_annotations())
        # Make sure that array annotations have the right length
        if cls in [IrregularlySampledSignal, AnalogSignal]:
            try:
                n = len(kwargs['signal'][0])
            # If only 1 signal, len(int) is called, this raises a TypeError
            except TypeError:
                n = 1
        elif cls in [SpikeTrain, Event, Epoch]:
            n = len(kwargs['times'])
        # Array annotate any DataObject
        if issubclass(cls, DataObject):
            new_seed = iseed + 1 if iseed is not None else iseed
            kwargs['array_annotations'] = get_fake_value('array_annotations', dict, seed=new_seed,
                                                         obj=cls, n=n)
        kwargs['seed'] = seed

    return kwargs


def get_annotations():
    '''
    Returns a dict containing the default values for annotations for
    a class from neo.core.
    '''
    return dict([(str(i), ann) for i, ann in enumerate(TEST_ANNOTATIONS)])


def fake_epoch(seed=None, n=1):
    """
    Create a fake Epoch.

    We use this separate function because the attributes of
    Epoch are not independent (must all have the same size)
    """
    kwargs = get_annotations()
    if seed is not None:
        np.random.seed(seed)
    size = np.random.randint(5, 15)
    for i, attr in enumerate(Epoch._necessary_attrs + Epoch._recommended_attrs):
        if seed is not None:
            iseed = seed + i
        else:
            iseed = None
        if attr[0] in ('times', 'durations', 'labels'):
            kwargs[attr[0]] = get_fake_value(*attr, seed=iseed, obj=Epoch, shape=size)
        else:
            kwargs[attr[0]] = get_fake_value(*attr, seed=iseed, obj=Epoch, n=n)
    kwargs['seed'] = seed
    obj = Epoch(**kwargs)
    return obj


def fake_neo(obj_type="Block", cascade=True, seed=None, n=1):
    '''
    Create a fake NEO object of a given type. Follows one-to-many
    and many-to-many relationships if cascade.

    n (default=1) is the number of child objects of each type will be created.
    In cases like segment.spiketrains, there will be more than this number
    because there will be n for each unit, of which there will be n for
    each channelindex, of which there will be n.
    '''

    if hasattr(obj_type, 'lower'):
        cls = class_by_name[obj_type]
    else:
        cls = obj_type
        obj_type = obj_type.__name__

    if cls is Epoch:
        obj = fake_epoch(seed=seed, n=n)
    else:
        kwargs = get_fake_values(obj_type, annotate=True, seed=seed, n=n)
        obj = cls(**kwargs)

    # if not cascading, we don't need to do any of the stuff after this
    if not cascade:
        return obj

    # this is used to signal other containers that they shouldn't duplicate
    # data
    if obj_type == 'Block':
        cascade = 'block'
    for i, childname in enumerate(getattr(obj, '_child_objects', [])):
        # we create a few of each class
        for j in range(n):
            if seed is not None:
                iseed = 10 * seed + 100 * i + 1000 * j
            else:
                iseed = None
            child = fake_neo(obj_type=childname, cascade=cascade, seed=iseed, n=n)
            child.annotate(i=i, j=j)

            # if we are creating a block and this is the object's primary
            # parent, don't create the object, we will import it from secondary
            # containers later
            if (cascade == 'block' and len(child._parent_objects) > 0
                    and obj_type != child._parent_objects[-1]):
                continue
            getattr(obj, _container_name(childname)).append(child)

    # need to manually create 'implicit' connections
    if obj_type == 'Block':
        # connect data objects to segment
        for i, chx in enumerate(obj.channel_indexes):
            for k, sigarr in enumerate(chx.analogsignals):
                obj.segments[k].analogsignals.append(sigarr)
            for k, sigarr in enumerate(chx.irregularlysampledsignals):
                obj.segments[k].irregularlysampledsignals.append(sigarr)
            for j, unit in enumerate(chx.units):
                for k, train in enumerate(unit.spiketrains):
                    obj.segments[k].spiketrains.append(train)
    # elif obj_type == 'ChannelIndex':
    #    inds = []
    #    names = []
    #    chinds = np.array([unit.channel_indexes[0] for unit in obj.units])
    #    obj.indexes = np.array(inds, dtype='i')
    #    obj.channel_names = np.array(names).astype('S')

    if hasattr(obj, 'create_many_to_one_relationship'):
        obj.create_many_to_one_relationship()

    return obj


def clone_object(obj, n=None):
    '''
    Generate a new object and new objects with the same rules as the original.
    '''
    if hasattr(obj, '__iter__') and not hasattr(obj, 'ndim'):
        return [clone_object(iobj, n=n) for iobj in obj]

    cascade = hasattr(obj, 'children') and len(obj.children)
    if n is not None:
        pass
    elif cascade:
        n = min(len(getattr(obj, cont)) for cont in obj._child_containers)
    else:
        n = 0
    seed = obj.annotations.get('seed', None)

    newobj = fake_neo(obj.__class__, cascade=cascade, seed=seed, n=n)
    if 'i' in obj.annotations:
        newobj.annotate(i=obj.annotations['i'], j=obj.annotations['j'])
    return newobj
