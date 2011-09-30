.. _use_cases_page:

*****************
Typical use cases
*****************





Multi segment recording
=======================

Note that this defines two ways you can access your data: in time, or in space. Each way involves descending the hierarchy of Neo objects a little differently.

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

Alternatively, instead of one big AnalogSignalArray in each Segment containing all channels, you could use one AnalogSignalArray for each RecordingChannelGroup. Different users find different structures to be more natural.




Spike sorting
=============


EEG
===



Network simulations
===================

