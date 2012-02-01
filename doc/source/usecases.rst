.. _use_cases_page:

*****************
Typical use cases
*****************





Multi segment recording for AnalogSIgnal
========================================

Here the typical example to illustrate a multi segment recording for signal.
There 1 Block composed by:

  * 3 Segment representing for instance trials.
  * 1 probe (RecordingChannelGroup) with 8 RecordingChannel

In a such case, you can finally play with 3x8 = 24 AnalogSignal. Named here as AS i,j (i = segment.index and j = recordingchannel.index)

.. image:: images/multi_segment_diagram.png

To summary, you can acces by 2 ways Segment or RecordingChannel that represent in this particulary example **time** by **space**.
Note that segments do not always represent trial, it can be use for many purpose: segment could represent parralels recordings for different subject
or segments could represent several steps in current clamp protocol.


**Temporal (by segment)**

In this case you want to go through your data in order, perhaps because you want to correlate the neural response with the stimulus that was delivered in each segment.
We'll assume that you've already put one trial per segment, and that the LFP data is in AnalogSignal. In this example, we won't worry about separating the response by channel::

    import numpy as np
    from matplotlib import pyplot as plt
    
    for seg in block.segments:
        print "Analyzing segment %d" % segment.index
        
        siglist = seg.analogsignals
        avg = np.mean(siglist, axis = 0)

        plt.figure()
        plt.plot(avg)
        plt.title("Peak response in segment %d: %f" % \
            (segment.index, avg.max()))

**Hardware (by channel)**

In this case you want to go through your data by channel location and average over time. 
Perhaps you want to see which physical location produces the strongest response, and every stimulus was the same::
    
    # We assume that our block have only 1 RecordingChannelGroup
    rcg = block.recordingchannelgroups[0]:
    for rc in rcg.recordingchannels:
        print "Analyzing channel %d: %s", (rc.index, rc.name)
        
        siglist = rc.analogsignals
        avg = np.mean(siglist, axis = 0)
        
        plt.figure()
        plt.plot(avg)
        plt.title("Average response on channel %d: %s' %(rc.index, rc.name)

Note that Block.list_recordingchannels is a proterty that acces directly to all RecordingChannel.
So the 2 first lines::

    rcg = block.recordingchannelgroups[0]:
    for rc in rcg.recordingchannels:

could be written as::
    
    for rc in block.list_recordingchannels:
        print "Analyzing channel %d: %s", (rc.index, rc.name)


**Mixed example**

Combining the two approaches of descending the hierarchy temporally and spatially simultaneously can be tricky. Here's an example.
Let's say you saw something interesting on channel 5 on even numbered trials during the experiment and you want to follow up. What was the average response?::
    
    avg = np.mean([seg.analogsignals[5] for seg in block.segments[::2]], axis=1)
    plt.plot(avg)

Note that in that example we assume that segment are ordered in a block.segments and analogsignals are also ardered in seg.analogsignals.
In a non symetric and with missing channel it could be safer do loop with testing index attribute for RecordingChannel and Segment. In this example,
note that we acces the segment througth the AnalogSignal by risingup the hierachy with many_to_one relationship::
    
    siglist = [ ]
    rcg = block.recordingchannelgroups[0]:
    for rc in rcg.recordingchannels:
        if rc.index == 5:
            for anasig in rc.analogsignals:
                if anasig.segment.index %2 == 0: # <-- here we use the many to one relationship.
                    siglist.append(anasig)
    avg = np.mean(siglist)
    



Multi segment recording for SpikeTrain
======================================

Here an equivalent example with SpikeTrain in a multi segment recording.

There 1 Block composed by:
  * 3 Segment representing for instance trials.
  * 2 RecordingChannelGroup (= tetrode in that case) with respectively:
  
    * 4 RecordingChannel for each
    * 2 Unit (= 2 neurons) for the frist RecordingChannelGroup
    * 5 Unit for the second RecordingChannelGroup

Note that there are 3x7=21 SpikeTrain in this Block.

.. image:: images/multi_segment_diagram_spiketrain.png

If you want to access SpikeTrain you have 3 possibilities:
  * by Segment. In our example it respresent trials.
  * by RecordingChannel. In this example, they are tetrode.
  * by Unit. It can be usefull if the location of the Neuron do not matter

**by Segment**

In this example, we assume that each Segment is a trial and we want a PSTH for each trial from of all Unit blend::

    for seg in block.segments:
        print "Analyzing segment %d" % segment.index
        
        stlist = [st - st.t_start for st in seg.spiketrains]
        plt.figure()
        count, bins = np.histogram(stlist)
        plt.bar(bins[:-1], count, width = bins[1] - bins[0])
        plt.title("PSTH in segment %d" % segment.index)

**by Unit**

In this example we want a PSTH average over trial for each Unit. Note that block.list_units is a property ::

    for unit in block.list_units:
        stlist = [st - st.t_start for st in unit.spiketrains]
        plt.figure()
        count, bins = np.histogram(stlist)
        plt.bar(bins[:-1], count, width = bins[1] - bins[0])
        plt.title("PSTH of unit %s" % unit.name)
        

**by RecordingChannelGroup**

In this example we want a PSTH average over trial by channel location blending all Unit (RecordingChannelGroup=tetrode in our case)::
    
    for rcg in blocl.recordingchannelgroups:
        stlist = [ ]
        for unit in rcg.units:
            stlist.append( [st - st.t_start for st in unit.spiketrains] )
        plt.figure()
        count, bins = np.histogram(stlist)
        plt.bar(bins[:-1], count, width = bins[1] - bins[0])
        plt.title("PSTH blend of RCG  %s" % rcg.name)


Spike sorting
=============


EEG
===



Network simulations
===================

