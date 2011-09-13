# encoding: utf-8
"""
This file is a bundle of utilities to discribe neo object representation (attributes and reslationships).

It can be used to:
 * generate diagrams of neo
 * some generics IO like (databases mapper, hdf5, neomatlab, ...)
 * tests
 * external SQL mappers (Cf OpenElectrophy, Gnode)


**classes_necessary_attributes**
This dict descibe attributes that are necessary.
It a dict of list of tuples.
Each attributes is describe by a tuple:
 * for standard type, the tuple is: (name + python type )
 * for np.ndarray type, the tuple is : (name + np.ndarray+ndim+dtype)
 * for pq.Quantities, the tuple is : (name+pq.Quantity+ndim)
ndim is the dimentionaly of the array 1=vector, 2=matrix, 3 = cube, ...
Special case ndim = 0, this mean that neo expect  a scalar so Quantity.shape=(1,)
that is in fact a vector (ndim=1) with one element only in Quantities package.

For some neo.object, the data is not nold by a field but by the object itself. This is the case
for AnalogSignal, SpikeTrain : they hinerit Quantity (that also hinerit numpy.array).
In theses case, an empty field (the first) in classes_necessary_attributes is added to describe
the object inheritence+ndim or dtype.


**classes_recommended_attributes**
This dict descibe attributes that are recommended.
Notation is same as classes_necessary_attributes.


"""

from .core import objectlist
 
import quantities as pq
from datetime import datetime
import numpy as np


class_by_name = { }
name_by_class = { }

for ob in objectlist:
    class_by_name[ob.__name__] = ob
    name_by_class[ob] = ob.__name__


# parent to children
one_to_many_reslationship = {
    'Block' : [ 'Segment','RecordingChannelGroup', ],
    'Segment' : [ 'AnalogSignal', 'AnalogSignalArray', 'IrregularlySampledSignal', 
                         'Event', 'EventArray', 'Epoch', 'EpochArray',
                        'SpikeTrain', 'Spike', ],
    'RecordingChannel' : [ 'AnalogSignal',  'IrregularlySampledSignal', ],
   'RecordingChannelGroup' : [ 'RecordingChannel',  'AnalogSignalArray'],
    'Unit' : ['SpikeTrain', 'Spike', ]
    }
# reverse: child to parent
many_to_one_reslationship = { }
for p,children in one_to_many_reslationship.iteritems():
    for c in children:
        if c not in many_to_one_reslationship:
            many_to_one_reslationship[c] = [ ]
        if p not in many_to_one_reslationship[c]:
            many_to_one_reslationship[c].append(p)

many_to_many_reslationship = {
    'RecordingChannel' : ['Unit', ],
    'Unit' : ['RecordingChannel', ],
    }
# check bijectivity
for p,children in many_to_many_reslationship.iteritems():
    for c in children:
        if c not in many_to_many_reslationship:
            many_to_many_reslationship[c] = [ ]
        if p not in many_to_many_reslationship[c]:
            many_to_many_reslationship[c].append[p]



classes_necessary_attributes = {
    'Block': [
                    ],
                    
    'Segment': [
                    ],
    
    'Event': [( 'time', pq.Quantity, 0 ),
                    ( 'label', str ),
                    ],
    
    'EventArray': [( 'times', pq.Quantity, 1 ),
                            ( 'labels',  np.ndarray,1 ,  np.dtype('S')) 
                            ],
    
    'Epoch': [ ( 'time', pq.Quantity, 0 ),
                    ( 'duration', pq.Quantity, 0 ),
                    ( 'label', str ),
                    ],
    
    'EpochArray': [( 'times', pq.Quantity, 1 ),
                            ( 'durations', pq.Quantity, 1 ),
                            ( 'labels',  np.ndarray,1,  np.dtype('S')) 
                            ],
    
    'Unit': [ ],
    
    'SpikeTrain': [('', pq.Quantity, 1 ),
                            ('t_start', pq.Quantity, 0 ),
                            ('t_stop', pq.Quantity, 0 ),
                            ],
    'Spike': [('time', pq.Quantity, 0),
                    ],
    
    'AnalogSignal': [('', pq.Quantity, 1 ),
                                ('sampling_rate', pq.Quantity, 0 ),
                                ('t_start', pq.Quantity, 0 ),
                                ],
    'AnalogSignalArray': [('', pq.Quantity, 2 ),
                                        ('sampling_rate', pq.Quantity, 0 ),
                                        ('t_start', pq.Quantity, 0 ),
                                        ],
    
    'IrregularlySampledSignal': [('samples',pq.Quantity,1),
                                        ('times',pq.Quantity,1),
                                    ],
    
   'RecordingChannelGroup': [ ],
    'RecordingChannel': [('index', int),
                                        ],
    }

classes_recommended_attributes= { 
    'Block': [( 'file_datetime', datetime ),
                    ( 'rec_datetime', datetime ),
                    ( 'index', int ), ],
    
    'Segment': [( 'file_datetime', datetime ),
                    ( 'rec_datetime', datetime ),
                    ( 'index', int ), ],
    
    'Event': [ ],
    'EventArray': [ ],
    'Epoch': [ ],
    'EpochArray': [ ],
    'Unit': [ ],
    'SpikeTrain': [('waveforms', pq.Quantity, 3),
                            ('left_sweep', pq.Quantity, 0 ),
                            ('sampling_rate', pq.Quantity, 0 ),
                            ],
    'Spike': [('waveform', pq.Quantity, 2),
                    ('left_sweep', pq.Quantity, 0 ),
                    ('sampling_rate', pq.Quantity, 0 ), ],
    'AnalogSignal': [
                                ],
    
    'AnalogSignalArray': [
                                        ],
    
    'IrregularlySampledSignal': [ 
                                                    ],
   'RecordingChannelGroup': [ ('channel_names', np.ndarray,1,  np.dtype('S')),
                                                ('channel_indexes', np.ndarray,1,  np.dtype('i')),
                                            ],
    'RecordingChannel': [('coordinate',pq.Quantity,1),],
    
    }

# all classes can have name, description, file_origin
for k in classes_recommended_attributes.keys():
    classes_recommended_attributes[k] += [ ('name', str ), ('description', str ), ('file_origin', str ),]



