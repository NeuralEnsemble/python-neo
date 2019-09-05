# -*- coding: utf-8 -*-


import inspect
from .baseio import BaseIO
from neo.core import ImageSequence, Segment, Block
import numpy as np


class AsciiImageIO(BaseIO):
    """
    Neo IO module for optical imaging data stored as a TXT file of images.
    """

    name = 'AsciiImage IO'
    description = "Neo IO module for optical imaging data stored as a folder of TIFF images."

    _prefered_signal_group_mode = 'group-by-same-units'
    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, ImageSequence]
    readable_objects = supported_objects
    writeable_object = []

    support_lazy = False

    read_params = {}
    write_params = {}

    extensions = []

    mode = 'file'

    def __init__(self, file_name=None, **kwargs):
        BaseIO.__init__(self, file_name, **kwargs)

    def read(self, lazy=False, units=None, sampling_rate=None, spatial_scale=None, **kwargs):
        if lazy:
            raise ValueError('This IO module does not support lazy loading')
        return [self.read_block(lazy=lazy, units=units, sampling_rate=sampling_rate,
                                spatial_scale=spatial_scale, **kwargs)]

    def read_block(self, lazy=False, nb_frame=None, nb_row=None, nb_column=None, units=None, sampling_rate=None,
                   spatial_scale=None, **kwargs):

        data = open(self.filename, 'r').read()

        liste_value = []
        record = []

        for i in range(len(data)):

            if data[i] == "\n" or data[i] == "\t":
                t = "".join(str(e) for e in record)
                liste_value.append(t)
                record = []
            else:
                record.append(data[i])

        data = []
        nb = 0
        for i in range(nb_frame):
            data.append([])
            for y in range(nb_row):
                data[i].append([])
                for x in range(nb_column):
                    data[i][y].append(liste_value[nb])
                    nb += 1

        print("read block")
        image_sequence = ImageSequence(np.array(data, dtype='float'), units=units,
                                       sampling_rate=sampling_rate, spatial_scale=spatial_scale)
        print("creating segment")
        segment = Segment(file_origin=self.filename)
        segment.imagesequences = [image_sequence]

        block = Block(file_origin=self.filename)
        segment.block = block
        block.segments.append(segment)
        print("returning block")

        return block