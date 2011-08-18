# encoding: utf-8
"""
This file is a bundle of utilities to discribe neo object representation (attributes and reslationships).

It can be used to:
 * generate diagrams of neo
 * some generics IO like (databases mapper, hdf5, ...)
 * tests
 * external SQL mappers (Cf OpenElectrophy, Gnode)


**classes_necessary_attributes**
This dict descibe attributes that are necessary.
It a dict of list of tuples.
Each attributes is describe by a tuple:
 * for standard type, the tuple is: (name + python type )
 * for np.array type, the tuple is : (name + np.array+ dtype+ndim)
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


TO BE DISCUSSED Proposal:
 * Unit > CellUnit
 * AnalogSignal.channel_index in classes_recommended_attributes or classes_necessary_attributes
 * SpikeTrain.waveforms_features for compatibility with KlustaquikIO in classes_recommended_attributes
"""

from .core import *
 
import quantities as pq
from datetime import datetime
import numpy as np

class_by_name = {
    'block': Block,
    'segment': Segment,
    'event': Event,
    'eventarray': EventArray,
    'epoch': Epoch,
    'epocharray': EpochArray,
    'unit': Unit,
    'spiketrain': SpikeTrain,
    'analogsignal': AnalogSignal,
    'analogsignalarray': AnalogSignalArray,
    'irsaanalogsignal': IrregularlySampledSignal,
    'spike': Spike,
    'recordingchannelgroup': RecordingChannelGroup,
    'recordingchannel': RecordingChannel,
    }

name_by_class = { }
for k,v in class_by_name.iteritems():
    name_by_class[v] = k

classnames = class_by_name



one_to_many_reslationship = {
    'block' : [ 'segment', 'recordingchannelgroup', ],
    'segment' : [ 'analogsignal', 'analogsignalarray', 'irsaanalogsignal', 
                         'event', 'eventarray', 'epoch', 'epocharray',
                        'spiketrain', 'spike', ],
    'recordingchannel' : [ ],
    'recordingchannelgroup' : [ 'recordingchannel',  'analogsignalarray'],
    'unit' : ['spiketrain', 'spike', ]
    }

many_to_many_reslationship = {
    'recordingchannel' : ['unit', ],
    'unit' : ['recordingchannel', ],
    }



classes_necessary_attributes = {
    'block': [
                    ],
                    
    'segment': [
                    ],
    
    'event': [( 'time', pq.Quantity, 0 ),
                    ( 'label', str ),
                    ],
    
    'eventarray': [( 'times', pq.Quantity, 1 ),
                            ( 'labels',  np.array, np.dtype('S'), 1) 
                            ],
    
    'epoch': [ ( 'time', pq.Quantity, 0 ),
                    ( 'duration', pq.Quantity, 0 ),
                    ( 'label', str ),
                    ],
    
    'epocharray': [( 'times', pq.Quantity, 1 ),
                            ( 'durations', pq.Quantity, 1 ),
                            ( 'labels',  np.array, np.dtype('S'), 1) 
                            ],
    
    'unit': [ ],
    
    'spiketrain': [('', pq.Quantity, 1 ),
                            ('t_start', pq.Quantity, 0 ),
                            ('t_stop', pq.Quantity, 0 ),
                            ],
    'spike': [('time', pq.Quantity, 0),
                    ],
    
    'analogsignal': [('', pq.Quantity, 1 ),
                                ('sampling_rate', pq.Quantity, 0 ),
                                ('t_start', pq.Quantity, 0 ),
                                ],
    'analogsignalarray': [('', pq.Quantity, 2 ),
                                        ('sampling_rate', pq.Quantity, 0 ),
                                        ('t_start', pq.Quantity, 0 ),
                                        ],
    
    'irsaanalogsignal': [('samples',pq.Quantity,1),
                                        ('times',pq.Quantity,1),
                                    ],
    
    'recordingchannelgroup': [ ],
    'recordingchannel': [('index', int),
                                        ],
    }

classes_recommended_attributes= { 
    'block': [( 'file_datetime', datetime ),
                    ( 'rec_datetime', datetime ),
                    ( 'index', int ), ],
    
    'segment': [( 'file_datetime', datetime ),
                    ( 'rec_datetime', datetime ),
                    ( 'index', int ), ],
    
    'event': [ ],
    'eventarray': [ ],
    'epoch': [ ],
    'epocharray': [ ],
    'unit': [ ],
    'spiketrain': [('waveforms', pq.Quantity, 3),
                            ('left_sweep', pq.Quantity, 0 ),
                            ('sampling_rate', pq.Quantity, 0 ),
                            ],
    'spike': [('waveform', pq.Quantity, 2),
                    ('left_sweep', pq.Quantity, 0 ),
                    ('sampling_rate', pq.Quantity, 0 ), ],
    'analogsignal': [('channel_name', str),
                                ('channel_index', int),
                                ],
    
    'analogsignalarray': [('channel_names', np.array, np.dtype('S'), 1),
                                        ('channel_indexes', np.array, np.dtype('i'),1),
                                        ],
    
    'irsaanalogsignal': [('channel_name', str),
                                        ('channel_index', int),
                                        ],
    'recordingchannelgroup': [ ],
    'recordingchannel': [('coordinate',pq.Quantity,1),],
    
    }

# main classes can have name, description, file_origin
for k in classes_recommended_attributes.keys():
    classes_recommended_attributes[k] += [ ('name', str ), ('description', str ) ]
for k in ['block', 'segment', 'spiketrain', 'analogsignal','analogsignalarray' ]:
    classes_recommended_attributes[k] += [  ('file_origin', str ), ]

