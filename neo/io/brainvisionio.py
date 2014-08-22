# -*- coding: utf-8 -*-
"""
Class for reading data from BrainVision product.

This code was originally made by L. Pezard (2010), modified B. Burle and S. More.

Supported : Read

Author: sgarcia
"""

import os
import re

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Segment, AnalogSignal, EventArray


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

        fmt = header['Binary Infos']['BinaryFormat']
        fmts = { 'INT_16':np.int16,  'INT_32':np.int32, 'IEEE_FLOAT_32':np.float32,}

        assert fmt in fmts, NotImplementedError
        dt = fmts[fmt]

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
            name, ref, res, units = header['Channel Infos']['Ch%d' % (c+1,)].split(',')
            units = pq.Quantity(1, units.replace('µ', 'u') )
            if lazy:
                signal = [ ]*units
            else:
                signal = sigs[:,c]*units
                if dt == np.int16 or dt == np.int32:
                    signal *= np.float(res) 
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
        all_info = readBrainSoup(marker_file)['Marker Infos']
        all_types = [ ]
        times = [ ]
        labels = [ ]
        for i in range(len(all_info)):
            type_, label, pos, size, channel = all_info['Mk%d' % (i+1,)].split(',')[:5]
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


        seg.create_many_to_one_relationship()
        return seg






def readBrainSoup(filename):
    section = None
    all_info = { }
    for line in open(filename , 'rU'):
        line = line.strip('\n').strip('\r')
        if line.startswith('['):
            section = re.findall('\[([\S ]+)\]', line)[0]
            all_info[section] = { }
            continue
        if line.startswith(';'):
            continue
        if '=' in line and len(line.split('=')) ==2:
            k,v = line.split('=')
            all_info[section][k] = v
    return all_info








