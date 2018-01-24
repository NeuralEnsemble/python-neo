"""Basic functionality for all data objects that inherit from pq.Quantity"""

import quantities as pq
import numpy as np

from neo.core.baseneo import BaseNeo, _check_annotations    # TODO: Deos this make sense? Should the _ be removed?


class DataObject(BaseNeo, pq.Quantity):

    # TODO: Do the _new_... functions also need to be changed?
    def __init__(self, name=None, description=None, file_origin=None, array_annotations=None,
                 **annotations):

        array_annotations = self._check_array_annotations(array_annotations)
        self.array_annotations = array_annotations

        BaseNeo.__init__(self, name=name, description=description, file_origin=file_origin, **annotations)

    # TODO: Okay to make it bound to instance instead of static like check_annotations?
    def _check_array_annotations(self, value):  # TODO: Is there anything else that can be checked here?

        # First stage, resolve dict of annotations into single annotations
        if isinstance(value, dict):
            for key in value.keys():
                if isinstance(value[key], dict):
                    raise ValueError("Dicts are not allowed as array annotations")  # TODO: Is this really the case?
                value[key] = self._check_annotations(value[key])

        # If not array annotation, pass on to regular check
        elif not isinstance(value, (list, np.ndarray)):
            _check_annotations(value)

        # If array annotation, check for correct length, only single dimension and
        else:
            try:
                own_length = self.shape[1]
            except IndexError:
                own_length = 1

            # Escape check if empty array or list
            if len(value) == 0:
                val_length = own_length
            else:
                # Note: len(o) also works for np.ndarray, it then uses the outmost dimension,
                # which is exactly the desired behaviour here
                val_length = len(value)

            if not own_length == val_length:
                raise ValueError("Incorrect length of array annotation")

            for element in value:
                if isinstance(element, (list, np.ndarray)):
                    raise ValueError("Array annotations should only be 1-dimensional")

                _check_annotations(value)

            if isinstance(value, list):
                value = np.array(value)

        return value

    def array_annotate(self, array_annotations):
        array_annotations = self._check_array_annotations(array_annotations)

    def array_annotations_at_index(self, index):  # TODO: Should they be sorted by key (current) or index?

        index_annotations = {}

        # Use what is given as an index to determine the corresponding annotations,
        # if not possible, numpy raises an Error
        for ann in self.annotations.keys():
            index_annotations[ann] = self.annotations[ann][index]

        return index_annotations

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
        Return the spike times as a quantities array.
        """
        return self.view(pq.Quantity)
