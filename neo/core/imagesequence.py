# -*- coding: utf-8 -*-
"""
This module implements :class:`ImageSequence`, a 3D array.

:class:`ImageSequence` inherits from :class:`basesignal.BaseSignal` which
derives from :class:`BaseNeo`, and from :class:`quantites.Quantity`which
in turn inherits from :class:`numpy.array`.

Inheritance from :class:`numpy.array` is explained here:
http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

In brief:
* Initialization of a new object from constructor happens in :meth:`__new__`.
This is where user-specified attributes are set.

* :meth:`__array_finalize__` is called for all new objects, including those
created by slicing. This is where attributes are copied over from
the old object.

"""

from neo.core.regionofinterest import RegionOfInterest
from neo.core.analogsignal import AnalogSignal, _get_sampling_rate

import quantities as pq
import numpy as np
from neo.core.baseneo import BaseNeo
from neo.core.basesignal import BaseSignal


class ImageSequence(BaseSignal):
    """
    Array of three dimension organize as [frame][row][column].

    Inherits from :class:`quantities.Quantity`, which in turn inherits from
    :class:`numpy.ndarray`.

    *usage*::

        >>> from neo.core import ImageSequence
        >>> import quantities as pq
        >>>
        >>> img_sequence_array = [[[column for column in range(20)]for row in range(20)]for frame in range(10)]
        >>> image_sequence = ImageSequence(img_sequence_array, units='V',
        ...                                sampling_rate=1*pq.Hz, spatial_scale=1*pq.micrometer)
        >>> image_sequence.all()
        ImageSequence

    *Required attributes/properties*:
        :image_data: (numpy array 3D, or list[frame][row][column]
            The data itself
        :units: (quantity units)
        :sampling_rate: *or* **sampling_period** (quantity scalar) Number of
                                                samples per unit time or
                                                interval beween to samples.
                                                If both are specified, they are
                                                checked for consistency.
        :spatial_scale: (quantity scalar) size for a pixel.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.

    *Optional attributes/properties*:
        :dtype: (numpy dtype or str) Override the dtype of the signal array.
        :copy: (bool) True by default.
        :array_annotations: (dict) Dict mapping strings to numpy
        arrays containing annotations for all data points

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`.

    *Properties available on this object*:
        :sampling_rate: (quantity scalar) Number of samples per unit time.
            (1/:attr:`sampling_period`)
        :sampling_period: (quantity scalar) Interval between two samples.
            (1/:attr:`quantity scalar`)
        :spatial_scales: size of a pixel
        
     """
    # format ImageSequence subclass dataobject
    # should be a 3d numerical array
    # format data[image_index][y][x]
    # meta data sampling interval/frame rate , spatia scale
    #
    # should contain a method  which take one or more regionofinterest as argument
    # and returns an analogicsignal
    #
    # exemples c2_avg  1 px =25ym  1 frame 2ms

    _single_parent_objects = ('Segment')
    _single_parent_attrs = ('segment')
    _quantity_attr = 'image_data'
    _necessary_attrs = (('image_data', pq.Quantity, 3),
                        ('sampling_rate', pq.Quantity, 0),
                        ('spatial_scale', pq.Quantity, 0))
    _recommended_attrs = BaseNeo._recommended_attrs

    def __new__(cls, image_data, units=None, dtype=None, copy=True, spatial_scale=None, sampling_period=None,
                sampling_rate=None, name=None, description=None, file_origin=None, array_annotations=None,
                **annotations):

        """
        Constructs new :class:`ImageSequence` from data.

        This is called whenever a new class:`ImageSequence` is created from
        the constructor, but not when slicing.

        __array_finalize__ is called on the new object.

        """

        if spatial_scale is None:
            raise ValueError('spatial_scale is required')
        if units == None:
            raise ValueError("units is required")

        image_data = np.stack(image_data)

        if len(image_data.shape) != 3:
            raise ValueError('list doesn\'t have the good number of dimension')

        obj = pq.Quantity(image_data, units=units, dtype=dtype, copy=copy).view(cls)
        obj.segment = None
        # function from analogsignal.py in neo/core directory
        obj.sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)
        obj.spatial_scale = spatial_scale

        return obj

    def __array_finalize__spec(self, obj):

        self.sampling_rate = getattr(obj, 'sampling_rate', None)
        self.spatial_scale = getattr(obj, 'spatial_scale', None)
        self.units = getattr(obj, 'units', None)

        return obj

    def signal_from_region(self, *region):


        if len(region) == 0:
            raise ValueError('no region of interest have been given')

        region_pixel = []
        for i, b in enumerate(region):
            r = region[i].return_list_pixel()
            if r == []:
                raise ValueError('region '+str(i)+'is empty')
            else:
                region_pixel.append(r)
        analogsignal_list = []
        for i in region_pixel:
            data = []
            for frame in range(len(self)):
                picture_data = []
                for v in i:
                    picture_data.append(self.view(pq.Quantity)[frame][v[0]][v[1]])
                average = picture_data[0]
                for b in range(1, len(picture_data)):
                    average += picture_data[b]
                data.append((average * 1.0) / len(i))
            analogsignal_list.append(AnalogSignal(data, units=self.units, sampling_rate=self.sampling_rate))

        return analogsignal_list
