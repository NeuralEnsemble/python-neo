"""
This module implements :class:`ChannelView`, which represents a subset of the
channels in an :class:`AnalogSignal` or :class:`IrregularlySampledSignal`.

It replaces the indexing function of the former :class:`ChannelIndex`.
"""

import numpy as np
from .baseneo import BaseNeo
from .basesignal import BaseSignal
from .dataobject import ArrayDict


class ChannelView(BaseNeo):
    """
    A tool for indexing a subset of the channels within an :class:`AnalogSignal`
    or :class:`IrregularlySampledSignal`;

    *Required attributes/properties*:
        :obj: (AnalogSignal or IrregularlySampledSignal) The signal being indexed.
        :index: (list/1D-array) boolean or integer mask to select the channels of interest.

    *Recommended attributes/properties*:
        :name: (str) A label for the view.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :array_annotations: (dict) Dict mapping strings to numpy arrays containing annotations
                            for all data points

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.
    """
    _parent_objects = ('Segment',)
    _parent_attrs = ('segment',)
    _necessary_attrs = (
        ('index', np.ndarray, 1, np.dtype('i')),
        ('obj', ('AnalogSignal', 'IrregularlySampledSignal'), 1)
    )
    # "mask" would be an alternative name, proposing "index" for
    # backwards-compatibility with ChannelIndex

    def __init__(self, obj, index, name=None, description=None, file_origin=None,
                 array_annotations=None, **annotations):
        super().__init__(name=name, description=description,
                         file_origin=file_origin, **annotations)

        if not (isinstance(obj, BaseSignal) or (
                hasattr(obj, "proxy_for") and issubclass(obj.proxy_for, BaseSignal))):
            raise ValueError("Can only take a ChannelView of an AnalogSignal "
                             "or an IrregularlySampledSignal")
        self.obj = obj

        # check type and dtype of index and convert index to a common form
        # (accept list or array of bool or int, convert to int array)
        self.index = np.array(index)
        if len(self.index.shape) != 1:
            raise ValueError("index must be a 1D array")
        if self.index.dtype == bool:  # convert boolean mask to integer index
            if self.index.size != self.obj.shape[-1]:
                raise ValueError("index size does not match number of channels in signal")
            self.index, = np.nonzero(self.index)
        # allow any type of integer representation
        elif self.index.dtype.char not in np.typecodes['AllInteger']:
            raise ValueError("index must be of a list or array of data type boolean or integer")

        if not hasattr(self, 'array_annotations') or not self.array_annotations:
            self.array_annotations = ArrayDict(self._get_arr_ann_length())
        if array_annotations is not None:
            self.array_annotate(**array_annotations)

    @property
    def shape(self):
        return (self.obj.shape[0], self.index.size)

    def _get_arr_ann_length(self):
        return self.shape[-1]

    def array_annotate(self, **array_annotations):
        self.array_annotations.update(array_annotations)

    def resolve(self):
        """
        Return a copy of the underlying object containing just the subset of channels
        defined by the index.
        """
        return self.obj[:, self.index]
