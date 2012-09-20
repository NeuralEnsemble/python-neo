# encoding: utf-8
"""
Generate datasets for testing

"""



from neo.core import *
from neo.io.tools import create_many_to_one_relationship, populate_RecordingChannel, iteritems
import numpy as np
import quantities as pq
from numpy.random import rand
from datetime import datetime

mV = pq.mV
uV = pq.uV
Hz = pq.Hz
kHz = pq.kHz
ms = pq.ms
nA = pq.nA
pA = pq.pA
s = pq.s


def generate_one_simple_block(block_name = 'block_0',
                                            nb_segment = 3, 
                                            supported_objects = [ ],
                                            **kws):
    bl = Block()#name = block_name)
    
    if Segment in supported_objects:
        for s in range(nb_segment):
            seg = generate_one_simple_segment(seg_name = "seg" + str(s),
                supported_objects = supported_objects, **kws)
            bl.segments.append(seg)
    
    if RecordingChannel in supported_objects:
        populate_RecordingChannel(bl)
    
    return bl


def generate_one_simple_segment(  seg_name = 'segment 0', 
                                                    supported_objects = [ ],
                                                    nb_analogsignal = 4,
                                                    t_start = 0.*s,
                                                    sampling_rate = 10*kHz,
                                                    duration = 6.*s,
                                                    
                                                    nb_spiketrain = 6,
                                                    spikerate_range = [.5*Hz, 12*Hz],
                                                    
                                                    event_array_types = {
                                                                                        'stim' : ['a', 'b', 'c' , 'd'],
                                                                                        'enter_zone' : [ 'one', 'two'],
                                                                                        'color' : ['black', 'yellow', 'green' ],
                                                                                        },
                                                    event_array_size_range = [5, 20],

                                                    epoch_array_types = {
                                                                                        'animal state' : ['Sleep', 'Freeze', 'Escape', ],
                                                                                        'light' : ['dark', 'lighted',  ],
                                                                                        },
                                                    epoch_array_duration_range = [.5, 3., ],

                                                ):
    seg = Segment(name = seg_name)
    if AnalogSignal in supported_objects:
        for a in range(nb_analogsignal):
            anasig = AnalogSignal( rand(int(sampling_rate*duration)), sampling_rate = sampling_rate, t_start = t_start,
                                  units=mV, name = 'sig %d for segment %s' % (a, seg.name) )
            anasig.annotations['channel_index'] = a
            seg.analogsignals.append(anasig)
    
    if SpikeTrain in supported_objects:
        for s in range(nb_spiketrain):
            spikerate = rand()*np.diff(spikerate_range)+spikerate_range[0].magnitude
            sptr = SpikeTrain( rand(int((spikerate*duration).simplified))*duration , t_start = t_start, t_stop = t_start+duration)
                                        #, name = 'spiketrain %d'%s)
            sptr.annotations['channel_index'] = s
            seg.spiketrains.append(sptr)
    
    if EventArray in supported_objects:
        for name, labels in iteritems(event_array_types):
            ea_size = rand()*np.diff(event_array_size_range)+event_array_size_range[0]
            ea = EventArray(     #name = name,
                                            times = rand(ea_size)*duration,
                                            labels = np.array( labels)[(rand(ea_size)*len(labels)).astype('i')],
                                            )
            seg.eventarrays.append(ea)
    
    if EpochArray in supported_objects:
        for name, labels in iteritems(epoch_array_types):
            t = 0
            times, durations = [ ], [ ]
            while t<duration:
                times.append(t)
                dur = (rand()*np.diff(epoch_array_duration_range)+epoch_array_duration_range[0])
                durations.append(dur)
                t = t+dur
            epa = EpochArray(    
                #name = name,
                times = pq.Quantity(times, units = pq.s),
                durations = pq.Quantity([x[0] for x in durations], units = pq.s),
                labels =  np.array( labels)[(rand(len(times))*len(labels)).astype('i')],
                )
            seg.epocharrays.append(epa)
            
    # TODO : Spike, Event, Epoch

    return seg




def generate_from_supported_objects( supported_objects ):
    #~ create_many_to_one_relationship
    if Block in supported_objects:
        higher = generate_one_simple_block(supported_objects= supported_objects)
        
        # Chris we do not create RC and RCG if it is not in supported_objects
        # there is a test in generate_one_simple_block so I removed
        #finalize_block(higher)
        
    elif Segment in supported_objects:
        higher = generate_one_simple_segment(supported_objects= supported_objects)
    else:
        #TODO
        return None
    
    create_many_to_one_relationship(higher)
    return higher
    
    


