*****************
Typical use cases
*****************

Recording multiple trials from multiple channels
================================================

In this example we suppose that we have recorded from an 8-channel probe, and
that we have recorded three trials/episodes. We therefore have a total of
8 x 3 = 24 signals, each represented by an :class:`AnalogSignal` object.

Our entire dataset is contained in a :class:`Block`, which in turn contains:

  * 3 :class:`Segment` objects, each representing data from a single trial,
  * 1 :class:`RecordingChannelGroup`, composed of 8 :class:`RecordingChannel` objects.

.. image:: images/multi_segment_diagram.png

:class:`Segment` and :class:`RecordingChannel` objects provide two different
ways to access the data, corresponding respectively, in this scenario, to access
by **time** and by **space**.

.. note:: segments do not always represent trials, they can be used for many
          purposes: segments could represent parallel recordings for different
          subjects, or different steps in a current clamp protocol.


**Temporal (by segment)**

In this case you want to go through your data in order, perhaps because you want
to correlate the neural response with the stimulus that was delivered in each segment.
In this example, we're averaging over the channels.

.. doctest::

    import numpy as np
    from matplotlib import pyplot as plt
    
    for seg in block.segments:
        print("Analyzing segment %d" % seg.index)
        
        siglist = seg.analogsignals
        avg = np.mean(siglist, axis=0)

        plt.figure()
        plt.plot(avg)
        plt.title("Peak response in segment %d: %f" % (seg.index, avg.max()))

**Spatial (by channel)**

In this case you want to go through your data by channel location and average over time. 
Perhaps you want to see which physical location produces the strongest response, and every stimulus was the same:
    
.. doctest::
    
    # We assume that our block has only 1 RecordingChannelGroup
    rcg = block.recordingchannelgroups[0]:
    for rc in rcg.recordingchannels:
        print("Analyzing channel %d: %s", (rc.index, rc.name))
        
        siglist = rc.analogsignals
        avg = np.mean(siglist, axis=0)
        
        plt.figure()
        plt.plot(avg)
        plt.title("Average response on channel %d: %s' % (rc.index, rc.name)

Note that :attr:`Block.list_recordingchannels` is a property that gives direct
access to all :class:`RecordingChannels`, so the two first lines::

    rcg = block.recordingchannelgroups[0]:
    for rc in rcg.recordingchannels:

could be written as::
    
    for rc in block.list_recordingchannels:


**Mixed example**

Combining simultaneously the two approaches of descending the hierarchy
temporally and spatially can be tricky. Here's an example.
Let's say you saw something interesting on channel 5 on even numbered trials
during the experiment and you want to follow up. What was the average response?

.. doctest::
    
    avg = np.mean([seg.analogsignals[5] for seg in block.segments[::2]], axis=1)
    plt.plot(avg)

Here we have assumed that segment are temporally ordered in a ``block.segments``
and that signals are ordered by channel number in ``seg.analogsignals``.
It would be safer, however, to avoid assumptions by explicitly testing the
:attr:`index` attribute of the :class:`RecordingChannel` and :class:`Segment`
objects. One way to do this is to loop over the recording channels and access
the segments through the signals (each :class:`AnalogSignal` contains a reference
to the :class:`Segment` it is contained in).
    
.. doctest::
    
    siglist = []
    rcg = block.recordingchannelgroups[0]:
    for rc in rcg.recordingchannels:
        if rc.index == 5:
            for anasig in rc.analogsignals:
                if anasig.segment.index % 2 == 0:
                    siglist.append(anasig)
    avg = np.mean(siglist)


Recording spikes from multiple tetrodes
=======================================

Here is a similar example in which we have recorded with two tetrodes and
extracted spikes from the extra-cellular signals. The spike times are contained
in :class:`SpikeTrain` objects.

Again, our data set is contained in a :class:`Block`, which contains:

  * 3 :class:`Segments` (one per trial).
  * 2 :class:`RecordingChannelGroups` (one per tetrode), which contain:
  
    * 4 :class:`RecordingChannels` each
    * 2 :class:`Unit` objects (= 2 neurons) for the first :class:`RecordingChannelGroup`
    * 5 :class:`Units` for the second :class:`RecordingChannelGroup`.

In total we have 3 x 7 = 21 :class:`SpikeTrains` in this :class:`Block`.

.. image:: images/multi_segment_diagram_spiketrain.png

There are three ways to access the :class:`SpikeTrain` data:

  * by :class:`Segment`
  * by :class:`RecordingChannel`
  * by :class:`Unit`

**By Segment**

In this example, each :class:`Segment` represents data from one trial, and we
want a PSTH for each trial from all units combined:

.. doctest::

    for seg in block.segments:
        print("Analyzing segment %d" % seg.index)
        stlist = [st - st.t_start for st in seg.spiketrains]
        plt.figure()
        count, bins = np.histogram(stlist)
        plt.bar(bins[:-1], count, width=bins[1] - bins[0])
        plt.title("PSTH in segment %d" % seg.index)

**By Unit**

Now we can calculate the PSTH averaged over trials for each unit, using the
:attr:`block.list_units` property:

.. doctest::

    for unit in block.list_units:
        stlist = [st - st.t_start for st in unit.spiketrains]
        plt.figure()
        count, bins = np.histogram(stlist)
        plt.bar(bins[:-1], count, width=bins[1] - bins[0])
        plt.title("PSTH of unit %s" % unit.name)
        

**By RecordingChannelGroup**

Here we calculate a PSTH averaged over trials by channel location,
blending all units:

.. doctest::

    for rcg in block.recordingchannelgroups:
        stlist = []
        for unit in rcg.units:
            stlist.extend([st - st.t_start for st in unit.spiketrains])
        plt.figure()
        count, bins = np.histogram(stlist)
        plt.bar(bins[:-1], count, width=bins[1] - bins[0])
        plt.title("PSTH blend of tetrode  %s" % rcg.name)


Spike sorting
=============


EEG
===



Network simulations
===================

