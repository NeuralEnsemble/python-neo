"""
Neo IO module for optical imaging data stored as a folder of TIFF images.
"""

import os

import numpy as np
from neo.core import ImageSequence, Segment, Block
from .baseio import BaseIO
import glob
import re


class TiffIO(BaseIO):
    """
    Neo IO module for optical imaging data stored as a folder of TIFF images.

    *Usage*:
        >>> from neo import io
        >>> import quantities as pq
        >>> r = io.TiffIO("dir_tiff",spatial_scale=1.0*pq.mm, units='V',
        ...               sampling_rate=1.0*pq.Hz)
        >>> block = r.read_block()
        read block
        creating segment
        returning block
        >>> block
        Block with 1 segments
        file_origin: 'test'
        # segments (N=1)
        0: Segment with 1 imagesequences
            annotations: {'tiff_file_names': ['file_tif_1_.tiff',
                'file_tif_2.tiff',
                'file_tif_3.tiff',
                'file_tif_4.tiff',
                'file_tif_5.tiff',
                'file_tif_6.tiff',
                'file_tif_7.tiff',
                'file_tif_8.tiff',
                'file_tif_9.tiff',
                'file_tif_10.tiff',
                'file_tif_11.tiff',
                'file_tif_12.tiff',
                'file_tif_13.tiff',
                'file_tif_14.tiff']}
            # analogsignals (N=0)
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

    def __init__(self, directory_path=None, units=None, sampling_rate=None,
                 spatial_scale=None, **kwargs):
        import PIL

        BaseIO.__init__(self, directory_path, **kwargs)
        self.units = units
        self.sampling_rate = sampling_rate
        self.spatial_scale = spatial_scale

    def read_block(self, lazy=False, **kwargs):
        import PIL

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
        # sorting file
        file_name_list = natural_sort(file_name_list)
        list_data_image = []
        for file_name in file_name_list:
            data = np.array(PIL.Image.open(self.filename + "/" + file_name)).astype(np.float32)
            list_data_image.append(data)
        list_data_image = np.array(list_data_image)
        if len(list_data_image.shape) == 4:
            list_data_image = []
            for file_name in file_name_list:
                image = PIL.Image.open(self.filename + "/" + file_name).convert('L')
                data = np.array(image).astype(np.float32)
                list_data_image.append(data)

        print("read block")
        image_sequence = ImageSequence(np.stack(list_data_image),
                                       units=self.units,
                                       sampling_rate=self.sampling_rate,
                                       spatial_scale=self.spatial_scale)
        print("creating segment")
        segment = Segment(file_origin=self.filename)
        segment.annotate(tiff_file_names=file_name_list)
        segment.imagesequences = [image_sequence]

        block = Block(file_origin=self.filename)
        segment.block = block
        block.segments.append(segment)
        print("returning block")
        return block
