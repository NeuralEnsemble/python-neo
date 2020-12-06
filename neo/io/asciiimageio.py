from .baseio import BaseIO
from neo.core import ImageSequence, Segment, Block
import numpy as np


class AsciiImageIO(BaseIO):
    """
    IO class for reading ImageSequence in a text file

    *Usage*:
        >>> from neo import io
        >>> import quantities as pq
        >>> r = io.AsciiImageIO(file_name='File_asciiimage_1.txt',nb_frame=511, nb_row=100,
        ...                     nb_column=100,units='mm', sampling_rate=1.0*pq.Hz,
        ...                     spatial_scale=1.0*pq.mm)
        >>> block = r.read_block()
        read block
        creating segment
        returning block
        >>> block
        Block with 1 segments
        file_origin: 'File_asciiimage_1.txt
        # segments (N=1)
        0: Segment with 1 imagesequences # analogsignals (N=0)

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

    def __init__(self, file_name=None, nb_frame=None, nb_row=None, nb_column=None, units=None, sampling_rate=None,
                 spatial_scale=None, **kwargs):

        BaseIO.__init__(self, file_name, **kwargs)
        self.nb_frame = nb_frame
        self.nb_row = nb_row
        self.nb_column = nb_column
        self.units = units
        self.sampling_rate = sampling_rate
        self.spatial_scale = spatial_scale

    def read_block(self, lazy=False, **kwargs):

        file = open(self.filename, 'r')
        data = file.read()
        print("read block")
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
        for i in range(self.nb_frame):
            data.append([])
            for y in range(self.nb_row):
                data[i].append([])
                for x in range(self.nb_column):
                    data[i][y].append(liste_value[nb])
                    nb += 1

        image_sequence = ImageSequence(np.array(data, dtype='float'), units=self.units,
                                       sampling_rate=self.sampling_rate, spatial_scale=self.spatial_scale)
        file.close()
        print("creating segment")
        segment = Segment(file_origin=self.filename)
        segment.imagesequences = [image_sequence]

        block = Block(file_origin=self.filename)
        segment.block = block
        block.segments.append(segment)
        print("returning block")

        return block
