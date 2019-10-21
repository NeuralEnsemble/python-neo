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

from neo.core.analogsignal import AnalogSignal, _get_sampling_rate
import quantities as pq
import numpy as np
from neo.core.baseneo import BaseNeo
from neo.core.basesignal import BaseSignal
from neo.core.dataobject import DataObject


class ImageSequence(BaseSignal):
    """
    Representation of a sequence of images, as an array of three dimensions
    organized as [frame][row][column].

    Inherits from :class:`quantities.Quantity`, which in turn inherits from
    :class:`numpy.ndarray`.

    *usage*::

        >>> from neo.core import ImageSequence
        >>> import quantities as pq
        >>>
        >>> img_sequence_array = [[[column for column in range(20)]for row in range(20)]
        ...                         for frame in range(10)]
        >>> image_sequence = ImageSequence(img_sequence_array, units='V',
        ...                                sampling_rate=1*pq.Hz, spatial_scale=1*pq.micrometer)
        >>> image_sequence
        ImageSequence 10 frame with 20 px of height and 20  px of width; units V; datatype int64
        sampling rate: 1.0
        spatial_scale: 1.0
        >>> image_sequence.spatial_scale
        array(1.) * um

    *Required attributes/properties*:
        :image_data: (3D NumPy array, or a list of 2D arrays)
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

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`.

    *Properties available on this object*:
        :sampling_rate: (quantity scalar) Number of samples per unit time.
            (1/:attr:`sampling_period`)
        :sampling_period: (quantity scalar) Interval between two samples.
            (1/:attr:`quantity scalar`)
        :spatial_scale: size of a pixel
     """
    _single_parent_objects = ('Segment',)
    _single_parent_attrs = ('segment',)
    _quantity_attr = 'image_data'
    _necessary_attrs = (('image_data', pq.Quantity, 3),
                        ('sampling_rate', pq.Quantity, 0),
                        ('spatial_scale', pq.Quantity, 0))
    _recommended_attrs = BaseNeo._recommended_attrs

    def __new__(cls, image_data, units=None, dtype=None, copy=True, spatial_scale=None, sampling_period=None,
                sampling_rate=None, name=None, description=None, file_origin=None,
                **annotations):
        """
        Constructs new :class:`ImageSequence` from data.

        This is called whenever a new class:`ImageSequence` is created from
        the constructor, but not when slicing.

        __array_finalize__ is called on the new object.
        """
        if spatial_scale is None:
            raise ValueError('spatial_scale is required')

        image_data = np.stack(image_data)
        if len(image_data.shape) != 3:
            raise ValueError('list doesn\'t have the good number of dimension')

        obj = pq.Quantity(image_data, units=units, dtype=dtype, copy=copy).view(cls)
        obj.segment = None
        # function from analogsignal.py in neo/core directory
        obj.sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)
        obj.spatial_scale = spatial_scale

        return obj

    def __init__(self, image_data, units=None, dtype=None, copy=True, spatial_scale=None, sampling_period=None,
                 sampling_rate=None, name=None, description=None, file_origin=None,
                 **annotations):
        '''
               Initializes a newly constructed :class:`ImageSequence` instance.
        '''
        DataObject.__init__(self, name=name, file_origin=file_origin, description=description,
                            **annotations)

    def __array_finalize__spec(self, obj):

        self.sampling_rate = getattr(obj, 'sampling_rate', None)
        self.spatial_scale = getattr(obj, 'spatial_scale', None)
        self.units = getattr(obj, 'units', None)

        return obj

    def signal_from_region(self, *region):
        """
            Method that takes 1 or multiple regionofinterest, use the method of each region
            of interest to get the list of pixel to average.
            return a list of :class:`AnalogSignal` for each regionofinterest
        """

        if len(region) == 0:
            raise ValueError('no region of interest have been given')

        region_pixel = []
        for i, b in enumerate(region):
            r = region[i].pixels_in_region()
            if not r:
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
            analogsignal_list.append(AnalogSignal(data, units=self.units,
                                                  sampling_rate=self.sampling_rate))

        return analogsignal_list

    def _repr_pretty_(self, pp, cycle):
        '''
               Handle pretty-printing the :class:`ImageSequence`.
        '''
        pp.text("{cls} {frame} frame with {width} px of width and {height} px of height; "
                "units {units}; datatype {dtype} ".format(cls=self.__class__.__name__,
                                                          frame=self.shape[0],
                                                          height=self.shape[1],
                                                          width=self.shape[2],
                                                          units=self.units.dimensionality.string,
                                                          dtype=self.dtype))

        def _pp(line):
            pp.breakable()
            with pp.group(indent=1):
                pp.text(line)

        for line in ["sampling rate: {0}".format(self.sampling_rate),
                     "spatial_scale: {0}".format(self.spatial_scale)]:
            _pp(line)

    def _check_consistency(self, other):
        '''
        Check if the attributes of another :class:`ImageSequence`
        are compatible with this one.
        '''
        if isinstance(other, ImageSequence):
            for attr in ("sampling_rate", "spatial_scale"):
                if getattr(self, attr) != getattr(other, attr):
                    raise ValueError("Inconsistent values of %s" % attr)
