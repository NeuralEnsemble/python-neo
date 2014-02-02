# -*- coding: utf-8 -*-
"""
This file is a bundle of utilities to describe Neo object representation
(attributes and relationships).

It can be used to:
 * generate diagrams of neo
 * some generics IO like (databases mapper, hdf5, neomatlab, ...)
 * tests
 * external SQL mappers (Cf OpenElectrophy, Gnode)


**classes_necessary_attributes**
This dict descibes attributes that are necessary to initialize an instance.
It a dict of list of tuples.
Each attribute is described by a tuple:
 * for standard type, the tuple is: (name + python type)
 * for np.ndarray type, the tuple is: (name + np.ndarray + ndim + dtype)
 * for pq.Quantities, the tuple is: (name + pq.Quantity + ndim)
ndim is the dimensionaly of the array: 1=vector, 2=matrix, 3=cube, ...
Special case: ndim=0 means that neo expects a scalar, so Quantity.shape=(1,).
That is in fact a vector (ndim=1) with one element only in Quantities package.

For some neo.object, the data is not held by a field, but by the object itself.
This is the case for AnalogSignal, SpikeTrain: they inherit from Quantity,
which itself inherits from numpy.array.
In theses cases, the classes_inheriting_quantities dict provide a list of
classes inhiriting Quantity and there attribute that will become Quantity
itself.


**classes_recommended_attributes**
This dict describes recommended attributes, which are optional at
initialization. If present, they will be stored as attributes of the object.
The notation is the same as classes_necessary_attributes.

"""

from datetime import datetime

import numpy as np
import quantities as pq

from neo.core import objectlist


class_by_name = {}
name_by_class = {}

for ob in objectlist:
    class_by_name[ob.__name__] = ob
    name_by_class[ob] = ob.__name__

classes_necessary_attributes = {
    'Block': [],

    'Segment': [],

    'Event': [('time', pq.Quantity, 0),
              ('label', str)],

    'EventArray': [('times', pq.Quantity, 1),
                   ('labels', np.ndarray, 1, np.dtype('S'))],

    'Epoch': [('time', pq.Quantity, 0),
              ('duration', pq.Quantity, 0),
              ('label', str)],

    'EpochArray': [('times', pq.Quantity, 1),
                   ('durations', pq.Quantity, 1),
                   ('labels', np.ndarray, 1, np.dtype('S'))],

    'Unit': [],

    'SpikeTrain': [('times', pq.Quantity, 1),
                   ('t_start', pq.Quantity, 0),
                   ('t_stop', pq.Quantity, 0)],
    'Spike': [('time', pq.Quantity, 0)],

    'AnalogSignal': [('signal', pq.Quantity, 1),
                     ('sampling_rate', pq.Quantity, 0),
                     ('t_start', pq.Quantity, 0)],

    'AnalogSignalArray': [('signal', pq.Quantity, 2),
                          ('sampling_rate', pq.Quantity, 0),
                          ('t_start', pq.Quantity, 0)],

    'IrregularlySampledSignal': [('times', pq.Quantity, 1),
                                 ('signal', pq.Quantity, 1)],

    'RecordingChannelGroup': [],

    'RecordingChannel': [('index', int)],
    }

classes_recommended_attributes = {
    'Block': [('file_datetime', datetime),
              ('rec_datetime', datetime),
              ('index', int), ],

    'Segment': [('file_datetime', datetime),
                ('rec_datetime', datetime),
                ('index', int)],

    'Event': [],

    'EventArray': [],

    'Epoch': [],

    'EpochArray': [],

    'Unit': [],

    'SpikeTrain': [('waveforms', pq.Quantity, 3),
                   ('left_sweep', pq.Quantity, 0),
                   ('sampling_rate', pq.Quantity, 0)],

    'Spike': [('waveform', pq.Quantity, 2),
              ('left_sweep', pq.Quantity, 0),
              ('sampling_rate', pq.Quantity, 0)],

    'AnalogSignal': [('channel_index', int)],

    'Unit': [('channel_indexes', np.ndarray, 1, np.dtype('i'))],

    'AnalogSignalArray': [('channel_index', np.ndarray, 1, np.dtype('i'))],

    'IrregularlySampledSignal': [],

    'RecordingChannelGroup': [('channel_indexes', np.ndarray,
                               1, np.dtype('i')),
                              ('channel_names', np.ndarray,
                               1, np.dtype('S'))],

    'RecordingChannel': [('coordinate', pq.Quantity, 1)],
    }

# this list classes inheriting quantities with arguments that will become
# the quantity array
classes_inheriting_quantities = {
    'SpikeTrain': 'times',
    'AnalogSignal': 'signal',
    'AnalogSignalArray': 'signal',
    'IrregularlySampledSignal':  'signal',
    }


# all classes can have name, description, file_origin
for k in classes_recommended_attributes.keys():
    classes_recommended_attributes[k] += [('name', str),
                                          ('description', str),
                                          ('file_origin', str)]
