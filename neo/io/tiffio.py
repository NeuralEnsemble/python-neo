 # -*- coding: utf-8 -*-
"""
Neo IO module for optical imaging data stored as a folder of TIFF images.

"""

import os
from PIL import Image
import numpy as np
from neo.core import ImageSequence, Segment, Block
from .baseio import BaseIO
import glob
import re

class TiffIO(BaseIO):
    """
    Neo IO module for optical imaging data stored as a folder of TIFF images.
    """
    name = 'TIFF IO'
    description = "Neo IO module for optical imaging data stored as a folder of TIFF images."

    _prefered_signal_group_mode = 'group-by-same-units'
    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, ImageSequence]
    readable_objects = supported_objects
    writeable_objects = []

    support_lazy = False

    read_params = {}
    write_params = {}

    extensions = []

    mode = 'dir'

    def __init__(self, directory_path=None, **kwargs):
        BaseIO.__init__(self, directory_path, **kwargs)

    def read(self, lazy=False, units=None, sampling_rate=None, spatial_scale=None, **kwargs):
        if lazy:
            raise ValueError('This IO module does not support lazy loading')
        return [self.read_block(lazy=lazy, units=units, sampling_rate=sampling_rate,
                                spatial_scale=spatial_scale, **kwargs)]

    def read_block(self, lazy=False, units=None, sampling_rate=None,
                   spatial_scale=None, **kwargs):
        # to sort file
        def natural_sort(l):
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key=alphanum_key)

        # find all the images in the given directory
        file_name_list = []
        # name of extensions to track
        types = ["*.tif", "*.tiff"]
        for file in types:
            file_name_list.append(glob.glob(self.filename+"/"+file))
        # flatten list
        file_name_list = [item for sublist in file_name_list for item in sublist]
        # delete path in the name of file
        file_name_list = [file_name[len(self.filename)+1::] for file_name in file_name_list]
        #sorting file
        file_name_list = natural_sort(file_name_list)
        list_data_image = []
        for file_name in file_name_list:
            list_data_image.append(np.array(Image.open(self.filename + "/" + file_name), dtype=np.float))
        list_data_image = np.array(list_data_image)
        if len(list_data_image.shape) == 4:
            list_data_image = []
            for file_name in file_name_list:
                list_data_image.append(np.array(Image.open(self.filename + "/" + file_name).convert('L'), dtype=np.float))

        print("read block")
        image_sequence = ImageSequence(np.stack(list_data_image),
                                       units=units,
                                       sampling_rate=sampling_rate,
                                       spatial_scale=spatial_scale)
        print("creating segment")
        segment = Segment(file_origin=self.filename)
        segment.annotate(tiff_file_names=file_name_list)
        segment.imagesequences = [image_sequence]

        block = Block(file_origin=self.filename)
        segment.block = block
        block.segments.append(segment)
        print("returning block")
        return block
