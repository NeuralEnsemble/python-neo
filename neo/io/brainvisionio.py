# encoding: utf-8
"""
Class for reading data from BrainVision product.

This code was originally made by L. Pezard (2010), modified B. Burle and S. More.



Supported : Read

Author: sgarcia





"""

from .baseio import BaseIO
from ..core import *
from .tools import create_many_to_one_relationship

import numpy as np
import quantities as pq

import os
import datetime
import re


class BrainVisionIO(BaseIO):
    """
    Class for reading/writing data from BrainVision product (brainAmp, brain analyser...)

    Usage:
        >>> from neo import io
        >>> r = io.BrainVisionIO( filename = 'File_brainvision_1.eeg')
        >>> seg = r.read_segment(lazy = False, cascade = True,)



    """

    is_readable        = True
    is_writable        = False

    supported_objects  = [Segment, AnalogSignal, EventArray]
    readable_objects   = [Segment]
    writeable_objects  = [ ]

    has_header         = False
    is_streameable     = False

    read_params        = { Segment : [ ] }
    write_params       = { Segment : [ ] }

    name               = None
    extensions         = ['vhdr']

    mode = 'file'


    def __init__(self , filename = None) :
        """
        This class read/write a elan based file.

        **Arguments**
            filename : the filename to read or write
        """
        BaseIO.__init__(self)
        self.filename = filename


    def read_segment(self, lazy = False, cascade = True):

        ## Read header file (vhdr)
        header = readBrainSoup(self.filename)

        assert header['Common Infos']['DataFormat'] == 'BINARY', NotImplementedError
        assert header['Common Infos']['DataOrientation'] == 'MULTIPLEXED', NotImplementedError
        nb_channel = int(header['Common Infos']['NumberOfChannels'])
        sampling_rate = 1.e6/float(header['Common Infos']['SamplingInterval']) * pq.Hz

        format = header['Binary Infos']['BinaryFormat']
        formats = { 'INT_16':np.int16, 'IEEE_FLOAT_32':np.float32,}
        assert format in formats, NotImplementedError
        dt = formats[format]

        seg = Segment(file_origin = os.path.basename(self.filename), )
        if not cascade : return seg

        # read binary
        if not lazy:
            binary_file = os.path.splitext(self.filename)[0]+'.eeg'
            sigs = np.memmap(binary_file , dt, 'r', ).astype('f')

            n = int(sigs.size/nb_channel)
            sigs = sigs[:n*nb_channel]
            sigs = sigs.reshape(n, nb_channel)

        for c in range(nb_channel):
            name, ref, res, units = header['Channel Infos']['Ch{}'.format(c+1)].split(',')
            units = pq.Quantity(1, units.replace('µ', 'u') )
            if lazy:
                signal = [ ]*units
            else:
                signal = sigs[:,c]*units
            anasig = AnalogSignal(signal = signal,
                                                channel_index = c,
                                                name = name,
                                                sampling_rate = sampling_rate,
                                                )
            if lazy:
                anasig.lazy_shape = -1
            seg.analogsignals.append(anasig)

        # read marker
        marker_file = os.path.splitext(self.filename)[0]+'.vmrk'
        all = readBrainSoup(marker_file)['Marker Infos']
        all_types = [ ]
        times = [ ]
        labels = [ ]
        for i in range(len(all)):
            type_, label, pos, size, channel = all['Mk{}'.format(i+1)].split(',')[:5]
            all_types.append(type_)
            times.append(float(pos)/sampling_rate.magnitude)
            labels.append(label)
        all_types = np.array(all_types)
        times = np.array(times) * pq.s
        labels = np.array(labels, dtype = 'S')
        for type_ in np.unique(all_types):
            ind = type_  == all_types
            if lazy:
                ea = EventArray(name = str(type_))
                ea.lazy_shape = -1
            else:
                ea = EventArray( times = times[ind],
                                    labels  = labels[ind],
                                    name = str(type_),
                                    )
            seg.eventarrays.append(ea)


        create_many_to_one_relationship(seg)
        return seg






def readBrainSoup(filename):
    section = None
    all = { }
    for line in open(filename , 'rU'):
        line = line.strip('\n').strip('\r')
        if line.startswith('['):
            section = re.findall('\[([\S ]+)\]', line)[0]
            all[section] = { }
            continue
        if line.startswith(';'):
            continue
        if '=' in line and len(line.split('=')) ==2:
            k,v = line.split('=')
            all[section][k] = v
    return all








