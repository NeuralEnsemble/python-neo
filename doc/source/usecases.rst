.. _use_cases_page:

*****************
Typical use cases
*****************





Multi segment recording
=======================

Here the typical example to illustrate a multi segment recording.
I have 1 Block:
  * with 3 Segment representing for instance trials.
  * with a probe with 8 RecordingChannel
In a such case, you can finally play with 3x8 = 24 AnalogSignal. Named here as AS i,j (i = segment.index and j = recordingchannel.index)

.. image:: images/multi_segment_diagram.png

To summary, you can acces by 2 ways Segment or RecordingChannel that represnt in this particulary example time or space.

Note that segments do not always represent trial, it can be use for many purpose but in this example block.segments.

.. todo:: continue thsi paragraph based on the schema



Temporal
In this case you want to go through your data in order, perhaps because you want to correlate the neural response with the stimulus that was delivered in each segment. We'll assume that you've already put one trial per segment, and that the LFP data is an AnalogSignalArray in the segment, and that the segment also contains spiketrains. In this example we won't worry about separating the response by channel or neuron.::

    for seg in block.segments:
        print "Analyzing segment %d" % segment.index
    
        # Get the average LFP across all channels
        resp1 = seg.analogsignalarrays[0].mean(axis=1)
        plt.figure()
        plt.plot(resp1)
        plt.title("Peak response in segment %d: %f" % \
            (segment.index, resp1.max()))
    
        # Get the average PSTH from all neurons during this segment
        stlist = [st - st.t_start for st in seg.spiketrains]
        plt.figure()
        mlab.hist(stlist)
        plt.title("PSTH in segment %d" % segment.index)

    plt.plot(np.mean(siglist))
    mlab.hist(np.concatenate(stlist))


* Hardware
In this case you want to go through your data by channel location and average over time. Perhaps you want to see which physical location produces the strongest response, and every stimulus was the same. We'll assume that you've assigned each tetrode in your dataset to be a RecordingChannelGroup, and you've sorted Units and Spiketrains from each neuron on each tetrode. (RecordingChannelGroup might also be a grid of electrodes over a certain region of the brain with EEG data.)::

    for rcg in block.recordingchannelgroups:
        print "Analyzing group %d: %s" (rcg.index, rcg.name)

        # Get the average EEG/LFP response across all channels in this group
        avg_resp = np.sum(rcg.analogsignalarrays)
        avg_resp = avg_resp / len(rcg.analogsignalarrays)
        plt.figure()
        plt.plot(avg_resp)
        plt.title('Average response on each channel of group %d' % rcg.index)

        # Plot average PSTH of each neuron in this group
        for neuron in rcg.units:
        stlist = [st - st.t_start for st in neuron.spiketrains]
        plt.figure()
        mlab.hist(stlist)
        plt.title("PSTH of unit %d on group %d" (neuron.index, rcg.index))


* Mixed Example
Combining the two approaches of descending the hierarchy temporally and spatially simultaneously can be tricky. Here's an example.

Let's say you saw something interesting on channel 5 on even numbered trials during the experiment and you want to follow up. What was the average response?::

    np.mean([seg.analogsignalarrays[0][:, 5].flatten()
        for seg in block.segments[::2]], axis=1)

Here we've assumed that each segment contains a single AnalogSignalArray which is somehow keyed by channel number, so that the 5th column is the 5th channel. If that's the case, it's best to label it with the annotations dict.::

    seg.analogsignalarrays[0].annotations['channel'] = np.arange(0, 10)
    print "channel 5 is in the %d-th column" % \
        np.nonzero(seg.analogsignalarrays[0].annotations['channel'] == 5)[0][0]

An alternative to the use of annotations is to crawl the hierarchy down in time and then up in hardware. Here we travel from an AnalogSignalArray to its hardware equivalent, a RecordingChannelGroup, and from an AnalogSignal to its hardware equivalent, a RecordingChannel.::

    rcg = seg.analogsignalarrays[0].recordingchannelgroup
    rc = seg.analogsignals[0].recordingchannel

This is possible because of the bijectivity of the many-to-one relationship, which is auto-created by neo.tools.create_many_to_one_relationship(temporal_object).

So to finish the example::

    # Which column to fetch from seg.analogsignalarrays[0]?
    colidx = seg.analogsignalarrays[0].recordingchannelgroup.\
        channel_indexes.index(5)
    data = seg.analogsignalarrays[0][:, 5]

What about neurons? The hardware equivalent of `SpikeTrain` is `Unit`, which is contained by `RecordingChannelGroup`. (Note: these connections are planned, not existing.)::

    # Get the spiketrains from unit #5 on even trials
    st_list = []
    for seg in block.segments[::2]:
        st_list.append(filter(lambda sptr: sptr.unit.index == 5 
            and sptr.unit.recordingchannelgroup.index == 0, seg.spiketrains))






Spike sorting
=============


EEG
===



Network simulations
===================

