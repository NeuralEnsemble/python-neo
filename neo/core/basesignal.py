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

import copy
import logging
from copy import deepcopy

import numpy as np
import quantities as pq

from neo.core.baseneo import MergeError, merge_annotations
from neo.core.dataobject import DataObject, ArrayDict

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
        super().__array_finalize__(obj)
        self._array_finalize_spec(obj)

        # The additional arguments
        self.annotations = getattr(obj, 'annotations', {})
        # Add empty array annotations, because they cannot always be copied,
        # but do not overwrite existing ones from slicing etc.
        # This ensures the attribute exists
        if not hasattr(self, 'array_annotations'):
            self.array_annotations = ArrayDict(self._get_arr_ann_length())

        # Globally recommended attributes
        self.name = getattr(obj, 'name', None)
        self.file_origin = getattr(obj, 'file_origin', None)
        self.description = getattr(obj, 'description', None)

        # Parent objects
        self.segment = getattr(obj, 'segment', None)

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
        f = getattr(super(), op)
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
            if attr[0] == "signal":
                required_attributes["signal"] = signal
            elif attr[0] == "image_data":
                required_attributes["image_data"] = signal
            elif attr[0] == "t_start":
                required_attributes["t_start"] = getattr(self, "t_start", 0.0 * pq.ms)
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
                if attr[0] == "t_start":
                    setattr(self, attr[0], deepcopy(getattr(other, attr[0], 0.0 * pq.ms)))
                elif attr[0] != 'signal':
                    setattr(self, attr[0], deepcopy(getattr(other, attr[0], None)))
        setattr(self, 'annotations', deepcopy(getattr(other, 'annotations', None)))

        # Note: Array annotations cannot be copied because length of data can be changed  # here
        #  which would cause inconsistencies

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
        stack = np.hstack((self.magnitude, other.magnitude))
        kwargs = {}
        for name in ("name", "description", "file_origin"):
            attr_self = getattr(self, name)
            attr_other = getattr(other, name)
            if attr_self == attr_other:
                kwargs[name] = attr_self
            else:
                kwargs[name] = "merge({}, {})".format(attr_self, attr_other)
        merged_annotations = merge_annotations(self.annotations, other.annotations)
        kwargs.update(merged_annotations)

        kwargs['array_annotations'] = self._merge_array_annotations(other)

        signal = self.__class__(stack, units=self.units, dtype=self.dtype, copy=False,
                                t_start=self.t_start, sampling_rate=self.sampling_rate, **kwargs)
        signal.segment = self.segment

        if hasattr(self, "lazy_shape"):
            signal.lazy_shape = merged_lazy_shape

        return signal

    def time_slice(self, t_start, t_stop):
        '''
        Creates a new AnalogSignal corresponding to the time slice of the
        original Signal between times t_start, t_stop.
        '''
        NotImplementedError('Needs to be implemented for subclasses.')

    def concatenate(self, *signals):
        '''
        Concatenate multiple signals across time.

        The signal objects are concatenated vertically
        (row-wise, :func:`np.vstack`). Concatenation can be
        used to combine signals across segments.
        Note: Only (array) annotations common to
        both signals are attached to the concatenated signal.

        If the attributes of the signals are not
        compatible, an Exception is raised.

        Parameters
        ----------
        signals : multiple neo.BaseSignal objects
            The objects that is concatenated with this one.

        Returns
        -------
        signal : neo.BaseSignal
            Signal containing all non-overlapping samples of
            the source signals.

        Raises
        ------
        MergeError
            If `other` object has incompatible attributes.
        '''

        NotImplementedError('Patching need to be implemented in subclasses')
