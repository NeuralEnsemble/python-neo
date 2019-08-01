from neo.core.regionofinterest import RegionOfInterest
from neo.core.analogsignal import AnalogSignal, _get_sampling_rate
from neo.core.dataobject import DataObject
import quantities as pq
import numpy as np
from neo.core.baseneo import BaseNeo


class ImageSequence(DataObject):
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

    def __new__(cls, image_data=None, units=None, dtype=None, copy=True, spatial_scale=None, sampling_period=None,
                sampling_rate=None, name=None, description=None, file_origin=None, array_annotations=None,
                **annotations):

        if spatial_scale is None:
            raise ValueError('spatial_scale is required')

        image_data = np.stack(image_data)

        if len(image_data.shape) != 3:
            raise ValueError('list doesn\'t have the good number of dimension')

        obj = pq.Quantity(image_data, units=units, dtype=dtype, copy=copy).view(cls)

        # function from analogsignal.py in neo/core directory
        obj.sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)
        obj.spatial_scale = spatial_scale

        return obj

    def __init__(self, image_data=None, units=None, sampling_rate=None, sampling_period=None, spatial_scale=None):

        self.__image_data = image_data
        self.__sampling_rate = sampling_rate
        self.__spatial_scale = spatial_scale
        self.__units = units

    def signal_from_region(self, *region):

        if len(region) == 0:
            raise ValueError('no region of interest have been given')

        region_pixel = []
        for i in range(len(region)):
            region_pixel.append(region[i].return_list_pixel())

        analogsignal_list = []
        for i in region_pixel:
            data = []
            for frame in range(len(self.__image_data)):
                picture_data = 0
                count = 0
                for v in i:
                    picture_data += self.__image_data[frame][v[0]][v[1]]
                    count += 1
                data.append((picture_data * 1.0) / count)
            analogsignal_list.append(AnalogSignal(data, self.units, sampling_rate=self.__sampling_rate))

        return analogsignal_list
