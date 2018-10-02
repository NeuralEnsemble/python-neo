# -*- coding: utf-8 -*-
"""
This module defines :class:`DataObject`, the abstract base class
used by all :module:`neo.core` classes that can contain data (i.e. are not container classes).
It contains basic functionality that is shared among all those data objects.

"""
import copy
import warnings

import quantities as pq
import numpy as np
from neo.core.baseneo import BaseNeo, _check_annotations


class DataObject(BaseNeo, pq.Quantity):
    '''
    This is the base class from which all objects containing data inherit
    It contains common functionality for all those objects and handles array_annotations.

    Common functionality that is not included in BaseNeo includes:
    - duplicating with new data
    - rescaling the object
    - copying the object
    - returning it as pq.Quantity or np.ndarray
    - handling of array_annotations

    Array_annotations are a kind of annotations that contain metadata for every data point,
    i.e. per timestamp (in SpikeTrain, Event and Epoch) or signal channel (in AnalogSignal
    and IrregularlySampledSignal).
    They can contain the same data types as regular annotations, but are always represented
    as numpy arrays of the same length as the number of data points of the annotated neo object.
    '''

    def __init__(self, name=None, description=None, file_origin=None, array_annotations=None,
                 **annotations):
        """
        This method is called from each data object and initializes the newly created object by
        adding array annotations and calling __init__ of the super class, where more annotations
        and attributes are processed.
        """

        # Adding array annotations to the object if not yet available, default is empty dict
        if array_annotations is None:
            if 'array_annotations' not in self.__dict__ or not self.array_annotations:
                self.array_annotations = {}
        else:
            self.array_annotate(**self._check_array_annotations(array_annotations))

        BaseNeo.__init__(self, name=name, description=description,
                         file_origin=file_origin, **annotations)

    def _check_array_annotations(self, value):

        """
        Recursively check that value is either an array or list containing only "simple" types
        (number, string, date/time) or is a dict of those.
        :return The array_annotations from value in correct form
        :raises ValueError: In case value is not accepted as array_annotation(s)
        """

        # First stage, resolve dict of annotations into single annotations
        if isinstance(value, dict):
            for key in value.keys():
                if isinstance(value[key], dict):
                    raise ValueError("Nested dicts are not allowed as array annotations")
                value[key] = self._check_array_annotations(value[key])

        elif value is None:
            raise ValueError("Array annotations must not be None")
        # If not array annotation, pass on to regular check and make it a list,
        # that is checked again
        # This covers array annotations with length 1
        elif not isinstance(value, (list, np.ndarray)) or \
                (isinstance(value, pq.Quantity) and value.shape == ()):
            _check_annotations(value)
            value = self._check_array_annotations(np.array([value]))

        # If array annotation, check for correct length, only single dimension and allowed data
        else:
            # Get length that is required for array annotations, which is equal to the length
            # of the object's data
            try:
                own_length = self._get_arr_ann_length()
            # FIXME This is because __getitem__[int] returns a scalar Epoch/Event/SpikeTrain
            # To be removed when __getitem__[int] is 'fixed'
            except IndexError:
                own_length = 1

            # Escape check if empty array or list and just annotate an empty array (length 0)
            # This enables the user to easily create dummy array annotations that will be filled
            # with data later on
            if len(value) == 0:
                if not isinstance(value, np.ndarray):
                    value = np.ndarray((0, ))
                val_length = own_length
            else:
                # Note: len(o) also works for np.ndarray, it then uses the outmost dimension,
                # which is exactly the desired behaviour here
                val_length = len(value)

            if not own_length == val_length:
                raise ValueError("Incorrect length of array annotation: {} != {}".
                                 format(val_length, own_length))

            # Local function used to check single elements of a list or an array
            # They must not be lists or arrays and fit the usual annotation data types
            def _check_single_elem(element):
                # Nested array annotations not allowed currently
                # So if an entry is a list or a np.ndarray, it's not allowed,
                # except if it's a quantity of length 1
                if isinstance(element, list) or \
                   (isinstance(element, np.ndarray) and not
                   (isinstance(element, pq.Quantity) and element.shape == ())):
                    raise ValueError("Array annotations should only be 1-dimensional")
                if isinstance(element, dict):
                    raise ValueError("Dicts are not supported array annotations")

                # Perform regular check for elements of array or list
                _check_annotations(element)

            # Arrays only need testing of single element to make sure the others are the same
            if isinstance(value, np.ndarray):
                # Type of first element is representative for all others
                # Thus just performing a check on the first element is enough
                # Even if it's a pq.Quantity, which can be scalar or array, this is still true
                # Because a np.ndarray cannot contain scalars and sequences simultaneously
                try:
                    # Perform check on first element
                    _check_single_elem(value[0])
                except IndexError:
                    # If length of data is 0, then nothing needs to be checked
                    pass
                return value

            # In case of list, it needs to be ensured that all data are of the same type
            else:
                # Check the first element for correctness
                # If its type is correct for annotations, all others are correct as well,
                # if they are of the same type
                # Note: Empty lists cannot reach this point
                _check_single_elem(value[0])
                dtype = type(value[0])

                # Loop through and check for varying datatypes in the list
                # Because these would create not clearly defined behavior
                # In case the user wants this, the list needs to be converted to np.ndarray first
                for element in value:
                    if not isinstance(element, dtype):
                        raise ValueError("Lists with different types are not supported for "
                                         "array annotations. ")

                # Create arrays from lists, because array annotations should be numpy arrays
                try:
                    value = np.array(value)
                except ValueError as e:
                    msg = str(e)
                    if "setting an array element with a sequence." in msg:
                        raise ValueError("Scalar Quantities and array Quanitities cannot be "
                                         "combined into a single array")
                    else:
                        raise e

        return value

    def array_annotate(self, **array_annotations):

        """
        Add annotations (non-standardized metadata) as arrays to a Neo data object.

        Example:

        >>> obj.array_annotate(code=['a', 'b', 'a'], category=[2, 1, 1])
        >>> obj.array_annotations['code'][1]
        'b'
        """

        array_annotations = self._check_array_annotations(array_annotations)
        self.array_annotations.update(array_annotations)

    def array_annotations_at_index(self, index):

        """
        Return dictionary of array annotations at a given index or list of indices
        :param index: int, list, numpy array: The index (indices) from which the annotations
                      are extracted
        :return: dictionary of values or numpy arrays containing all array annotations
                 for given index

        Example:
        >>> obj.array_annotate(code=['a', 'b', 'a'], category=[2, 1, 1])
        >>> obj.array_annotations_at_index(1)
        {code='b', category=1}
        """

        index_annotations = {}

        # Use what is given as an index to determine the corresponding annotations,
        # if not possible, numpy raises an Error
        for ann in self.array_annotations.keys():
            # NO deepcopy, because someone might want to alter the actual object using this
            try:
                index_annotations[ann] = self.array_annotations[ann][index]
            except IndexError as e:
                # IndexError caused by 'dummy' array annotations should not result in failure
                # Taking a slice from nothing results in nothing
                if len(self.array_annotations[ann]) == 0 and not self._get_arr_ann_length() == 0:
                    index_annotations[ann] = self.array_annotations[ann]
                else:
                    raise e

        return index_annotations

    def _merge_array_annotations(self, other):
        '''
        Merges array annotations of 2 different objects.
        The merge happens in such a way that the result fits the merged data
        In general this means concatenating the arrays from the 2 objects.
        If an annotation is only present in one of the objects, it will be omitted
        :return Merged array_annotations
        '''

        # Make sure the user is notified for every object about which exact annotations are lost
        warnings.simplefilter('always', UserWarning)
        merged_array_annotations = {}
        omitted_keys_self = []
        # Concatenating arrays for each key
        for key in self.array_annotations:
            try:
                value = copy.deepcopy(self.array_annotations[key])
                other_value = copy.deepcopy(other.array_annotations[key])
                # Quantities need to be rescaled to common unit
                if isinstance(value, pq.Quantity):
                    try:
                        other_value = other_value.rescale(value.units)
                    except ValueError:
                        raise ValueError("Could not merge array annotations "
                                         "due to different units")
                    merged_array_annotations[key] = np.append(value, other_value)*value.units
                else:
                    merged_array_annotations[key] = np.append(value, other_value)

            except KeyError:
                # Save the  omitted keys to be able to print them
                omitted_keys_self.append(key)
                continue
        # Also save omitted keys from 'other'
        omitted_keys_other = [key for key in other.array_annotations
                              if key not in self.array_annotations]
        # Warn if keys were omitted
        if omitted_keys_other or omitted_keys_self:
            warnings.warn("The following array annotations were omitted, because they were only "
                          "present in one of the merged objects: {} from the one that was merged "
                          "into and {} from the one that was merged into the other".
                          format(omitted_keys_self, omitted_keys_other), UserWarning)
        
        # Reset warning filter to default state
        warnings.simplefilter("default")

        # Return the merged array_annotations
        return merged_array_annotations

    def rescale(self, units):
        '''
        Return a copy of the object converted to the specified
        units
        :return: Copy of self with specified units
        '''
        # Use simpler functionality, if nothing will be changed
        dim = pq.quantity.validate_dimensionality(units)
        if self.dimensionality == dim:
            return self.copy()

        # Rescale the object into a new object
        # Works for all objects currently
        obj = self.duplicate_with_new_data(signal=self.view(pq.Quantity).rescale(dim),
                                           units=units)

        # Expected behavior is deepcopy, so deepcopying array_annotations
        obj.array_annotations = copy.deepcopy(self.array_annotations)

        obj.segment = self.segment

        return obj

    # Needed to implement this so array annotations are copied as well, ONLY WHEN copying 1:1
    def copy(self, **kwargs):
        '''
        Returns a copy of the object
        :return: Copy of self
        '''

        obj = super(DataObject, self).copy(**kwargs)
        obj.array_annotations = self.array_annotations
        return obj

    def as_array(self, units=None):
        """
        Return the object's data as a plain NumPy array.

        If `units` is specified, first rescale to those units.
        """
        if units:
            return self.rescale(units).magnitude
        else:
            return self.magnitude

    def as_quantity(self):
        """
        Return the object's data as a quantities array.
        """
        return self.view(pq.Quantity)

    def _get_arr_ann_length(self):
        """
        Return the length of the object's data as required for array annotations
        This is the last dimension of every object.
        :return Required length of array annotations for this object
        """
        # Number of items is last dimension in current objects
        # This holds true for the current implementation
        # This method should be overridden in case this changes
        return self.shape[-1]
