'''
Generate datasets for testing
'''

from datetime import datetime
import random
import string
import numpy as np
from numpy.random import rand
import quantities as pq

from neo.core import (AnalogSignal, Block, Epoch, Event, IrregularlySampledSignal, Group,
                      Segment, SpikeTrain, ImageSequence, ChannelView,
                      CircularRegionOfInterest, RectangularRegionOfInterest,
                      PolygonRegionOfInterest)

TEST_ANNOTATIONS = [1, 0, 1.5, "this is a test", datetime.fromtimestamp(424242424), None]


def random_string(length=10):
    return "".join(random.choice(string.ascii_letters) for i in range(length))


def random_datetime(min_year=1990, max_year=datetime.now().year):
    start = datetime(min_year, 1, 1, 0, 0, 0)
    end = datetime(max_year, 12, 31, 23, 59, 59)
    return start + (end - start) * random.random()


def random_annotations(n=1):
    annotation_generators = (
        random.random,
        random_datetime,
        random_string,
        lambda: None
    )
    annotations = {}
    for i in range(n):
        var_name = random_string(6)
        annotation_generator = random.choice(annotation_generators)
        annotations[var_name] = annotation_generator()
    return annotations


def random_signal(name=None, **annotations):
    n_channels = random.randint(1, 7)
    sig_length = random.randint(20, 200)
    if len(annotations) == 0:
        annotations = random_annotations(5)
    obj = AnalogSignal(
        np.random.uniform(size=(sig_length, n_channels)),
        units=random.choice(("mV", "nA")),
        t_start=random.uniform(0, 10) * pq.ms,
        sampling_rate=random.uniform(0.1, 10) * pq.kHz,
        name=name or random_string(),
        file_origin=random_string(),
        description=random_string(100),
        array_annotations=None,   # todo
        **annotations
    )
    return obj


def random_irreg_signal(name=None, **annotations):
    n_channels = random.randint(1, 7)
    sig_length = random.randint(20, 200)
    if len(annotations) == 0:
        annotations = random_annotations(5)
    mean_firing_rate = np.random.uniform(0.1, 10) * pq.kHz
    times = np.cumsum(np.random.uniform(1.0 / mean_firing_rate, size=(sig_length,))) * pq.ms
    obj = IrregularlySampledSignal(
        times,
        np.random.uniform(size=(sig_length, n_channels)),
        units=random.choice(("mV", "nA")),
        name=name or random_string(),
        file_origin=random_string(),
        description=random_string(100),
        array_annotations=None,   # todo
        **annotations
    )
    return obj


def random_event(name=None, **annotations):
    size = random.randint(1, 7)
    times = np.cumsum(np.random.uniform(5, 10, size=size))
    labels = [random_string() for i in range(size)]
    if len(annotations) == 0:
        annotations = random_annotations(3)
    obj = Event(
        times=times,
        labels=labels,
        units="ms",
        name=name or random_string(),
        array_annotations=None,   # todo
        **annotations
    )
    return obj


def random_epoch():
    size = random.randint(1, 7)
    times = np.cumsum(np.random.uniform(5, 10, size=size))
    durations = np.random.uniform(1, 3, size=size)
    labels = [random_string() for i in range(size)]
    obj = Epoch(
        times=times,
        durations=durations,
        labels=labels,
        units="ms",
        name=random_string(),
        array_annotations=None,   # todo
        **random_annotations(3)
    )
    return obj


def random_spiketrain(name=None, **annotations):
    size = random.randint(1, 50)
    times = np.cumsum(np.random.uniform(0.5, 10, size=size))
    if len(annotations) == 0:
        annotations = random_annotations(3)
    # todo: waveforms
    obj = SpikeTrain(
        times=times,
        t_stop=times[-1] + random.uniform(0, 5),
        units="ms",
        name=name or random_string(),
        array_annotations=None,   # todo
        **annotations
    )
    return obj


def random_segment():
    seg = Segment(
        name=random_string(10),
        description=random_string(100),
        file_origin=random_string(20),
        file_datetime=random_datetime(),
        rec_datetime=random_datetime(),
        **random_annotations(4)
    )
    n_sigs = random.randint(0, 5)
    for i in range(n_sigs):
        seg.analogsignals.append(random_signal())
    n_irrsigs = random.randint(0, 5)
    for i in range(n_irrsigs):
        seg.irregularlysampledsignals.append(random_irreg_signal())
    n_events = random.randint(0, 3)
    for i in range(n_events):
        seg.events.append(random_event())
    n_epochs = random.randint(0, 3)
    for i in range(n_epochs):
        seg.epochs.append(random_epoch())
    n_spiketrains = random.randint(0, 20)
    for i in range(n_spiketrains):
        seg.spiketrains.append(random_spiketrain())
    # todo: add some ImageSequence and ROI objects

    for child in seg.data_children:
        child.segment = seg
    return seg


def random_group(candidates):
    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        objects = candidates
    else:
        k = random.randint(1, len(candidates))
        objects = random.sample(candidates, k)
    obj = Group(objects=objects,
                name=random_string(),
                **random_annotations(5))
    return obj


def random_channelview(signal):
    n_channels = signal.shape[1]
    if n_channels > 2:
        view_size = np.random.randint(1, n_channels - 1)
        index = np.random.choice(np.arange(signal.shape[1]), view_size, replace=False)
        obj = ChannelView(
            signal,
            index,
            name=random_string(),
            **random_annotations(3)
        )
        return obj
    else:
        return None


def random_block():
    block = Block(
        name=random_string(10),
        description=random_string(100),
        file_origin=random_string(20),
        file_datetime=random_datetime(),
        rec_datetime=random_datetime(),
        **random_annotations(6)
    )
    n_seg = random.randint(0, 5)
    for i in range(n_seg):
        seg = random_segment()
        block.segments.append(seg)
        seg.block = block
    children = list(block.data_children_recur)
    views = []
    for child in children:
        if isinstance(child, (AnalogSignal, IrregularlySampledSignal)):
            PROB_SIGNAL_HAS_VIEW = 0.5
            if np.random.random_sample() < PROB_SIGNAL_HAS_VIEW:
                chv = random_channelview(child)
                if chv:
                    views.append(chv)
    children.extend(views)
    n_groups = random.randint(0, 5)
    for i in range(n_groups):
        group = random_group(children)
        if group:
            block.groups.append(group)
            group.block = block
            children.append(group)  # this can give us nested groups
    return block


def simple_block():
    block = Block(
        name="test block",
        species="rat",
        brain_region="cortex"
    )
    block.segments = [
        Segment(name="test segment #1",
                cell_type="spiny stellate"),
        Segment(name="test segment #2",
                cell_type="pyramidal",
                thing="amajig")
    ]
    for segment in block.segments:
        segment.block = block
    block.segments[0].analogsignals.extend((
        random_signal(name="signal #1 in segment #1", thing="wotsit"),
        random_signal(name="signal #2 in segment #1", thing="frooble"),
    ))
    block.segments[1].analogsignals.extend((
        random_signal(name="signal #1 in segment #2", thing="amajig"),
    ))
    block.segments[1].irregularlysampledsignals.extend((
        random_irreg_signal(name="signal #1 in segment #2", thing="amajig"),
    ))
    block.segments[0].events.extend((
        random_event(name="event array #1 in segment #1", thing="frooble"),
    ))
    block.segments[1].events.extend((
        random_event(name="event array #1 in segment #2", thing="wotsit"),
    ))
    block.segments[0].spiketrains.extend((
        random_spiketrain(name="spiketrain #1 in segment #1", thing="frooble"),
        random_spiketrain(name="spiketrain #2 in segment #1", thing="wotsit")
    ))
    return block


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
            anasig = AnalogSignal(rand(int((sampling_rate * duration).simplified)),
                                  sampling_rate=sampling_rate,
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
            labels = np.array(labels, dtype='U')
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
            labels = np.array(labels, dtype='U')
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
    elif Segment in objects:
        higher = generate_one_simple_segment(supported_objects=objects)
    else:
        # TODO
        return None

    higher.create_many_to_one_relationship()
    return higher
