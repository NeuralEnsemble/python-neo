*************************
Grouping and linking data
*************************


Migrating from ChannelIndex/Unit to ChannelView/Group
=====================================================

While the basic hierarchical :class:`Block` - :class:`Segment` structure of Neo has remained
unchanged since the inception of Neo, the structures used to cross-link objects
(for example to link a signal to the spike trains derived from it) have undergone changes,
in an effort to find an easily understandable and usable approach.

Below we give some examples of how to migrate from :class:`ChannelIndex` and :class:`Unit`,
as used in Neo 0.8, to the new classes :class:`Group` and :class:`ChannelView`
introduced in Neo 0.9.
Note that Neo 0.9 supports the new and old API in parallel, to facilitate migration.
IO classes in Neo 0.9 can read :class:`ChannelIndex` and :class:`Unit` objects,
but do not write them.

:class:`ChannelIndex` and :class:`Unit` will be removed in Neo 0.10.0.

Examples
--------

A simple example with two tetrodes. Here the :class:`ChannelIndex` was not being
used for grouping, simply to associate a name with each channel.

Using :class:`ChannelIndex`::

    import numpy as np
    from quantities import kHz, mV
    from neo import Block, Segment, ChannelIndex, AnalogSignal

    block = Block()
    segment = Segment()
    segment.block = block
    block.segments.append(segment)

    for i in (0, 1):
        signal = AnalogSignal(np.random.rand(1000, 4) * mV,
                              sampling_rate=1 * kHz,)
        segment.analogsignals.append(signal)
        chx = ChannelIndex(name=f"Tetrode #{i + 1}",
                           index=[0, 1, 2, 3],
                           channel_names=["A", "B", "C", "D"])
        chx.analogsignals.append(signal)
        block.channel_indexes.append(chx)

Using array annotations, we annotate the channels of the :class:`AnalogSignal` directly::

    import numpy as np
    from quantities import kHz, mV
    from neo import Block, Segment, AnalogSignal

    block = Block()
    segment = Segment()
    segment.block = block
    block.segments.append(segment)

    for i in (0, 1):
        signal = AnalogSignal(np.random.rand(1000, 4) * mV,
                              sampling_rate=1 * kHz,
                              channel_names=["A", "B", "C", "D"])
        segment.analogsignals.append(signal)


Now a more complex example: a 1x4 silicon probe, with a neuron on channels 0,1,2 and another neuron on channels 1,2,3.
We create a :class:`ChannelIndex` for each neuron to hold the :class:`Unit` object associated with this spike sorting group.
Each :class:`ChannelIndex` also contains the list of channels on which that neuron spiked.

::

    import numpy as np
    from quantities import ms, mV, kHz
    from neo import Block, Segment, ChannelIndex, Unit, SpikeTrain, AnalogSignal

    block = Block(name="probe data")
    segment = Segment()
    segment.block = block
    block.segments.append(segment)

    # create 4-channel AnalogSignal with dummy data
    signal = AnalogSignal(np.random.rand(1000, 4) * mV,
                          sampling_rate=10 * kHz)
    # create spike trains with dummy data
    # we will pretend the spikes have been extracted from the dummy signal
    spiketrains = [
        SpikeTrain(np.arange(5, 100) * ms, t_stop=100 * ms),
        SpikeTrain(np.arange(7, 100) * ms, t_stop=100 * ms)
    ]
    segment.analogsignals.append(signal)
    segment.spiketrains.extend(spiketrains)
    # assign each spiketrain to a neuron (Unit)
    units = []
    for i, spiketrain in enumerate(spiketrains):
        unit = Unit(name=f"Neuron #{i + 1}")
        unit.spiketrains.append(spiketrain)
        units.append(unit)

    # create a ChannelIndex for each unit, to show which channels the spikes come from
    chx0 = ChannelIndex(name="Channel Group 1", index=[0, 1, 2])
    chx0.units.append(units[0])
    chx0.analogsignals.append(signal)
    units[0].channel_index = chx0
    chx1 = ChannelIndex(name="Channel Group 2", index=[1, 2, 3])
    chx1.units.append(units[1])
    chx1.analogsignals.append(signal)
    units[1].channel_index = chx1

    block.channel_indexes.extend((chx0, chx1))


Using :class:`ChannelView` and :class`Group`::

    import numpy as np
    from quantities import ms, mV, kHz
    from neo import Block, Segment, ChannelView, Group, SpikeTrain, AnalogSignal

    block = Block(name="probe data")
    segment = Segment()
    segment.block = block
    block.segments.append(segment)

    # create 4-channel AnalogSignal with dummy data
    signal = AnalogSignal(np.random.rand(1000, 4) * mV,
                          sampling_rate=10 * kHz)
    # create spike trains with dummy data
    # we will pretend the spikes have been extracted from the dummy signal
    spiketrains = [
        SpikeTrain(np.arange(5, 100) * ms, t_stop=100 * ms),
        SpikeTrain(np.arange(7, 100) * ms, t_stop=100 * ms)
    ]
    segment.analogsignals.append(signal)
    segment.spiketrains.extend(spiketrains)
    # assign each spiketrain to a neuron (now using Group)
    units = []
    for i, spiketrain in enumerate(spiketrains):
        unit = Group(spiketrain, name=f"Neuron #{i + 1}")
        units.append(unit)

    # create a ChannelView of the signal for each unit, to show which channels the spikes come from
    # and add it to the relevant Group
    view0 = ChannelView(signal, index=[0, 1, 2], name="Channel Group 1")
    units[0].add(view0)
    view1 = ChannelView(signal, index=[1, 2, 3], name="Channel Group 2")
    units[1].add(view1)

    block.groups.extend(units)


Now each putative neuron is represented by a :class:`Group` containing the spiktrains of that neuron
and a view of the signal selecting only those channels from which the spikes were obtained.