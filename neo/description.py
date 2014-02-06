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


**classes_recommended_attributes**
This dict describes recommended attributes, which are optional at
initialization. If present, they will be stored as attributes of the object.
The notation is the same as classes_necessary_attributes.

"""

from datetime import datetime

import numpy as np
import quantities as pq


classes_necessary_attributes = {}

classes_recommended_attributes = {}
