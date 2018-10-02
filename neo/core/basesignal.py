# -*- coding: utf-8 -*-
'''
This module implements :class:`BaseSignal`, an array of signals.
This is a parent class from which all signal objects inherit:
    :class:`AnalogSignal` and :class:`IrregularlySampledSignal`

:class:`BaseSignal` inherits from :class:`quantities.Quantity`, which
inherits from :class:`numpy.array`.
Inheritance from :class:`numpy.array` is explained here:
http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

In brief:
* Constructor :meth:`__new__` for :class:`BaseSignal` doesn't exist. 
Only child objects :class:`AnalogSignal` and :class:`IrregularlySampledSignal`
can be created.
'''

# needed for Python 3 compatibility
from __future__ import absolute_import, division, print_function

import copy
import logging

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, MergeError, merge_annotations
from neo.core.dataobject import DataObject
from neo.core.channelindex import ChannelIndex

logger = logging.getLogger("Neo")


class BaseSignal(DataObject):
    '''
    This is the base class from which all signal objects inherit:
    :class:`AnalogSignal` and :class:`IrregularlySampledSignal`.

    This class contains all common methods of both child classes.
    It uses the following child class attributes:

        :_necessary_attrs: a list of the attributes that the class must have. 

        :_recommended_attrs: a list of the attributes that the class may 
        optionally have.
    '''

    def _array_finalize_spec(self, obj):
        '''
        Called by :meth:`__array_finalize__`, used to customize behaviour of sub-classes.
        '''
        return obj

    def __array_finalize__(self, obj):
        '''
        This is called every time a new signal is created.

        It is the appropriate place to set default values for attributes
        for a signal constructed by slicing or viewing.

        User-specified values are only relevant for construction from
        constructor, and these are set in __new__ in the child object.
        Then they are just copied over here. Default values for the 
        specific attributes for subclasses (:class:`AnalogSignal`
        and :class:`IrregularlySampledSignal`) are set in 
        :meth:`_array_finalize_spec`
        '''
        super(BaseSignal, self).__array_finalize__(obj)
        self._array_finalize_spec(obj)

        # The additional arguments
        self.annotations = getattr(obj, 'annotations', {})
        # Add empty array annotations, because they cannot always be copied,
        # but do not overwrite existing ones from slicing etc.
        # This ensures the attribute exists
        if not hasattr(self, 'array_annotations'):
            self.array_annotations = {}

        # Globally recommended attributes
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)

        # Parent objects
        self.segment = getattr(obj, 'segment', None)
        self.channel_index = getattr(obj, 'channel_index', None)

    @classmethod
    def _rescale(self, signal, units=None):
        '''
        Check that units are present, and rescale the signal if necessary.
        This is called whenever a new signal is
        created from the constructor. See :meth:`__new__' in  
        :class:`AnalogSignal` and :class:`IrregularlySampledSignal`
        '''
        if units is None:
            if not hasattr(signal, "units"):
                raise ValueError("Units must be specified")
        elif isinstance(signal, pq.Quantity):
            # This test always returns True, i.e. rescaling is always executed if one of the units
            # is a pq.CompoundUnit. This is fine because rescaling is correct anyway.
            if pq.quantity.validate_dimensionality(units) != signal.dimensionality:
                signal = signal.rescale(units)
        return signal

    def rescale(self, units):
        obj = super(BaseSignal, self).rescale(units)
        obj.channel_index = self.channel_index
        return obj

    def __getslice__(self, i, j):
        '''
        Get a slice from :attr:`i` to :attr:`j`.attr[0]

        Doesn't get called in Python 3, :meth:`__getitem__` is called instead
        '''
        return self.__getitem__(slice(i, j))

    def __ne__(self, other):
        '''
        Non-equality test (!=)
        '''
        return not self.__eq__(other)

    def _apply_operator(self, other, op, *args):
        '''
        Handle copying metadata to the new signal
        after a mathematical operation.
        '''
        self._check_consistency(other)
        f = getattr(super(BaseSignal, self), op)
        new_signal = f(other, *args)
        new_signal._copy_data_complement(self)
        # _copy_data_complement can't always copy array annotations,
        # so this needs to be done locally
        new_signal.array_annotations = copy.deepcopy(self.array_annotations)
        return new_signal

    def _get_required_attributes(self, signal, units):
        '''
        Return a list of the required attributes for a signal as a dictionary
        '''
        required_attributes = {}
        for attr in self._necessary_attrs:
            if 'signal' == attr[0]:
                required_attributes[str(attr[0])] = signal
            else:
                required_attributes[str(attr[0])] = getattr(self, attr[0], None)
        required_attributes['units'] = units
        return required_attributes

    def duplicate_with_new_data(self, signal, units=None):
        '''
        Create a new signal with the same metadata but different data.
        Required attributes of the signal are used.
        Note: Array annotations can not be copied here because length of data can change
        '''
        if units is None:
            units = self.units
        # else:
        #     units = pq.quantity.validate_dimensionality(units)

        # signal is the new signal
        required_attributes = self._get_required_attributes(signal, units)
        new = self.__class__(**required_attributes)
        new._copy_data_complement(self)
        new.annotations.update(self.annotations)
        # Note: Array annotations are not copied here, because it is not ensured
        # that the same number of signals is used and they would possibly make no sense
        # when combined with another signal
        return new

    def _copy_data_complement(self, other):
        '''
        Copy the metadata from another signal.
        Required and recommended attributes of the signal are used.
        Note: Array annotations can not be copied here because length of data can change
        '''
        all_attr = {self._recommended_attrs, self._necessary_attrs}
        for sub_at in all_attr:
            for attr in sub_at:
                if attr[0] != 'signal':
                    setattr(self, attr[0], getattr(other, attr[0], None))
        setattr(self, 'annotations', getattr(other, 'annotations', None))

        # Note: Array annotations cannot be copied because length of data can be changed
        # here which would cause inconsistencies

    def __rsub__(self, other, *args):
        '''
        Backwards subtraction (other-self)
        '''
        return self.__mul__(-1, *args) + other

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

    def merge(self, other):
        '''
        Merge another signal into this one.

        The signal objects are concatenated horizontally
        (column-wise, :func:`np.hstack`).

        If the attributes of the two signal are not
        compatible, an Exception is raised.

        Required attributes of the signal are used.
        '''

        for attr in self._necessary_attrs:
            if 'signal' != attr[0]:
                if getattr(self, attr[0], None) != getattr(other, attr[0], None):
                    raise MergeError("Cannot merge these two signals as the %s differ." % attr[0])

        if self.segment != other.segment:
            raise MergeError(
                "Cannot merge these two signals as they belong to different segments.")
        if hasattr(self, "lazy_shape"):
            if hasattr(other, "lazy_shape"):
                if self.lazy_shape[0] != other.lazy_shape[0]:
                    raise MergeError("Cannot merge signals of different length.")
                merged_lazy_shape = (self.lazy_shape[0], self.lazy_shape[1] + other.lazy_shape[1])
            else:
                raise MergeError("Cannot merge a lazy object with a real object.")
        if other.units != self.units:
            other = other.rescale(self.units)
        stack = np.hstack(map(np.array, (self, other)))
        kwargs = {}
        for name in ("name", "description", "file_origin"):
            attr_self = getattr(self, name)
            attr_other = getattr(other, name)
            if attr_self == attr_other:
                kwargs[name] = attr_self
            else:
                kwargs[name] = "merge(%s, %s)" % (attr_self, attr_other)
        merged_annotations = merge_annotations(self.annotations,
                                               other.annotations)
        kwargs.update(merged_annotations)

        kwargs['array_annotations'] = self._merge_array_annotations(other)

        signal = self.__class__(stack, units=self.units, dtype=self.dtype,
                                copy=False, t_start=self.t_start,
                                sampling_rate=self.sampling_rate,
                                **kwargs)
        signal.segment = self.segment

        if hasattr(self, "lazy_shape"):
            signal.lazy_shape = merged_lazy_shape

        # merge channel_index (move to ChannelIndex.merge()?)
        if self.channel_index and other.channel_index:
            signal.channel_index = ChannelIndex(
                index=np.arange(signal.shape[1]),
                channel_ids=np.hstack([self.channel_index.channel_ids,
                                       other.channel_index.channel_ids]),
                channel_names=np.hstack([self.channel_index.channel_names,
                                         other.channel_index.channel_names]))
        else:
            signal.channel_index = ChannelIndex(index=np.arange(signal.shape[1]))

        return signal
