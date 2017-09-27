# -*- coding: utf-8 -*-
'''
This module implements :class:`BaseSignal`, an array of signals.

:class:`BaseSignal` inherits from :class:`quantites.Quantity`, which
inherits from :class:`numpy.array`.
Inheritance from :class:`numpy.array` is explained here:
http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

In brief:
* Initialization of a new object from constructor happens in :meth:`__new__`.
This is where user-specified attributes are set.

* :meth:`__array_finalize__` is called for all new objects, including those
created by slicing. This is where attributes are copied over from
the old object.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, MergeError, merge_annotations
from neo.core.channelindex import ChannelIndex

logger = logging.getLogger("Neo")

class BaseSignal(BaseNeo, pq.Quantity):    
    
    def __ne__(self, other):
        '''
        Non-equality test (!=)
        '''
        return not self.__eq__(other)

    def _apply_operator(self, other, op, *args):
        '''
        Handle copying metadata to the new :class:`BaseSignal`
        after a mathematical operation.
        '''
        self._check_consistency(other)
        f = getattr(super(BaseSignal, self), op)
        new_signal = f(other, *args)
        new_signal._copy_data_complement(self)
        return new_signal

    def __add__(self, other, *args):
        '''
        Addition (+)
        '''
        return self._apply_operator(other, "__add__", *args)

    def __sub__(self, other, *args):
        '''
        Subtraction (-)
        '''
        return self._apply_operator(other, "__sub__", *args)

    def __mul__(self, other, *args):
        '''
        Multiplication (*)
        '''
        return self._apply_operator(other, "__mul__", *args)

    def __truediv__(self, other, *args):
        '''
        Float division (/)
        '''
        return self._apply_operator(other, "__truediv__", *args)

    def __div__(self, other, *args):
        '''
        Integer division (//)
        '''
        return self._apply_operator(other, "__div__", *args)

    __radd__ = __add__
    __rmul__ = __sub__

    def as_array(self, units=None):
        """
        Return the signal as a plain NumPy array.

        If `units` is specified, first rescale to those units.
        """
        if units:
            return self.rescale(units).magnitude
        else:
            return self.magnitude

    def as_quantity(self):
        """
        Return the signal as a quantities array.
        """
        return self.view(pq.Quantity)
