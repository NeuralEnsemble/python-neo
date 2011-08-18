# encoding: utf-8
"""
Generate datasets for testing

"""



from ...core import *
import numpy as np
import quantities as pq
from scipy import rand
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
    bl = Block(name = block_name)
    
    if Segment in supported_objects:
        for s in range(nb_segment):
            seg = generate_one_simple_segment(supported_objects=supported_objects, **kws)
            bl._segments.append(seg)
        
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
                                                    
                                                    
                                                ):
    seg = Segment(name= seg_name)
    if AnalogSignal in supported_objects:
        for a in range(nb_analogsignal):
            anasig = AnalogSignal( rand(int(sampling_rate*duration)), sampling_rate = sampling_rate, 
                                        t_start = t_start, name = 'sig %d og segment %s'%(a, seg.name),)
            seg._analogsignals.append(anasig)
    
    if SpikeTrain in supported_objects:
        for s in range(nb_spiketrain):
            spikerate = rand()*np.diff(spikerate_range)+spikerate_range[0].magnitude
            sptr = SpikeTrain( rand(int((spikerate*duration).simplified))*duration , t_start = t_start, t_stop = t_start+duration)
            seg._spiketrains.append(sptr)
    
    if EventArray in supported_objects:
        for name, labels in event_array_types.iteritems():
            ea_size = rand()*np.diff(event_array_size_range)+event_array_size_range[0]
            ea = EventArray(     name = name,
                                            times = rand(ea_size)*duration,
                                            labels = np.array( labels)[(rand(ea_size)*len(labels)).astype('i')],
                                            )
            seg._eventarrays.append(ea)
                                        
    
    # TODO other objects
    
    return seg




def generate_from_supported_objects( supported_objects ):
    if Block in supported_objects:
        higher = generate_one_simple_block(supported_objects= supported_objects)
    elif Segment in supported_objects:
        higher = generate_one_simple_segment(supported_objects= supported_objects)
    else:
        #TODO
        pass
    
    return higher
    
    


