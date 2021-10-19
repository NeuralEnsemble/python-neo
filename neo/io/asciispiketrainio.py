"""
Classe for reading/writing SpikeTrains in a text file.
It is the simple case where different spiketrains are written line by line.

Supported : Read/Write

Author: sgarcia

"""

import os

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Segment, SpikeTrain


class AsciiSpikeTrainIO(BaseIO):
    """

    Class for reading/writing SpikeTrains in a text file.
    Each Spiketrain is a line.

    Usage:
        >>> from neo import io
        >>> r = io.AsciiSpikeTrainIO( filename = 'File_ascii_spiketrain_1.txt')
        >>> seg = r.read_segment()
        >>> print seg.spiketrains     # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<SpikeTrain(array([ 3.89981604,  4.73258781,  0.608428  ,  4.60246277,  1.23805797,
        ...

    """

    is_readable = True
    is_writable = True

    supported_objects = [Segment, SpikeTrain]
    readable_objects = [Segment]
    writeable_objects = [Segment]

    has_header = False
    is_streameable = False

    read_params = {
        Segment: [
            ('delimiter', {'value': '\t', 'possible': ['\t', ' ', ',', ';']}),
            ('t_start', {'value': 0., }),
        ]
    }
    write_params = {
        Segment: [
            ('delimiter', {'value': '\t', 'possible': ['\t', ' ', ',', ';']}),
        ]
    }

    name = None
    extensions = ['txt']

    mode = 'file'

    def __init__(self, filename=None):
        """
        This class read/write SpikeTrains in a text file.
        Each row is a spiketrain.

        **Arguments**

        filename : the filename to read/write

        """
        BaseIO.__init__(self)
        self.filename = filename

    def read_segment(self,
                     lazy=False,
                     delimiter='\t',
                     t_start=0. * pq.s,
                     unit=pq.s,
                     ):
        """
        Arguments:
            delimiter  :  columns delimiter in file  '\t' or one space or two space or ',' or ';'
            t_start : time start of all spiketrain 0 by default
            unit : unit of spike times, can be a str or directly a Quantities
        """
        assert not lazy, 'Do not support lazy'

        unit = pq.Quantity(1, unit)

        seg = Segment(file_origin=os.path.basename(self.filename))

        with open(self.filename, 'r', newline=None) as f:
            for i, line in enumerate(f):
                alldata = line[:-1].split(delimiter)
                if alldata[-1] == '':
                    alldata = alldata[:-1]
                if alldata[0] == '':
                    alldata = alldata[1:]

                spike_times = np.array(alldata).astype('f')
                t_stop = spike_times.max() * unit

                sptr = SpikeTrain(spike_times * unit, t_start=t_start, t_stop=t_stop)

                sptr.annotate(channel_index=i)
                seg.spiketrains.append(sptr)

        seg.create_many_to_one_relationship()
        return seg

    def write_segment(self, segment,
                      delimiter='\t',
                      ):
        """
        Write SpikeTrain of a Segment in a txt file.
        Each row is a spiketrain.

         Arguments:
            segment : the segment to write. Only analog signals will be written.
            delimiter  :  columns delimiter in file  '\t' or one space or two space or ',' or ';'

            information of t_start is lost

        """

        f = open(self.filename, 'w')
        for s, sptr in enumerate(segment.spiketrains):
            for ts in sptr:
                f.write('{:f}{}'.format(ts, delimiter))
            f.write('\n')
        f.close()
