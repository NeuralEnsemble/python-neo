# -*- coding: utf-8 -*-
"""
Class for reading/writing data from micromed (.trc).
Inspired by the Matlab code for EEGLAB from Rami K. Niazy.

Completed with matlab Guillaume BECQ code.

Supported : Read

Author: sgarcia
"""

import datetime
import os
import struct

# file no longer exists in Python3
try:
    file
except NameError:
    import io
    file = io.BufferedReader

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Segment, AnalogSignal, EpochArray, EventArray


class struct_file(file):
    def read_f(self, fmt):
        return struct.unpack(fmt , self.read(struct.calcsize(fmt)))


class MicromedIO(BaseIO):
    """
    Class for reading  data from micromed (.trc).

    Usage:
        >>> from neo import io
        >>> r = io.MicromedIO(filename='File_micromed_1.TRC')
        >>> seg = r.read_segment(lazy=False, cascade=True)
        >>> print seg.analogsignals              # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<AnalogSignal(array([ -1.77246094e+02,  -2.24707031e+02,  -2.66015625e+02,
        ...
    """
    is_readable        = True
    is_writable        = False

    supported_objects            = [ Segment , AnalogSignal , EventArray, EpochArray ]
    readable_objects    = [Segment]
    writeable_objects    = [ ]

    has_header         = False
    is_streameable     = False
    read_params        = { Segment : [ ] }
    write_params       = None

    name               = None
    extensions          = [ 'TRC' ]

    mode = 'file'

    def __init__(self , filename = None) :
        """
        This class read a micromed TRC file.

        Arguments:
            filename : the filename to read
        """
        BaseIO.__init__(self)
        self.filename = filename


    def read_segment(self, cascade = True, lazy = False,):
        """
        Arguments:
        """
        f = struct_file(self.filename, 'rb')

        #Name
        f.seek(64,0)
        surname = f.read(22)
        while surname[-1] == ' ' :
            if len(surname) == 0 :break
            surname = surname[:-1]
        firstname = f.read(20)
        while firstname[-1] == ' ' :
            if len(firstname) == 0 :break
            firstname = firstname[:-1]

        #Date
        f.seek(128,0)
        day, month, year, hour, minute, sec = f.read_f('bbbbbb')
        rec_datetime = datetime.datetime(year+1900 , month , day, hour, minute, sec)

        f.seek(138,0)
        Data_Start_Offset , Num_Chan , Multiplexer , Rate_Min , Bytes = f.read_f('IHHHH')
        #~ print Num_Chan, Bytes

        #header version
        f.seek(175,0)
        header_version, = f.read_f('b')
        assert header_version == 4

        seg = Segment(  name = firstname+' '+surname,
                                    file_origin = os.path.basename(self.filename),
                                    )
        seg.annotate(surname = surname)
        seg.annotate(firstname = firstname)
        seg.annotate(rec_datetime = rec_datetime)

        if not cascade:
            return seg

        # area
        f.seek(176,0)
        zone_names = ['ORDER', 'LABCOD', 'NOTE', 'FLAGS', 'TRONCA', 'IMPED_B', 'IMPED_E', 'MONTAGE',
                'COMPRESS', 'AVERAGE', 'HISTORY', 'DVIDEO', 'EVENT A', 'EVENT B', 'TRIGGER']
        zones = { }
        for zname in zone_names:
            zname2, pos, length = f.read_f('8sII')
            zones[zname] = zname2, pos, length
            #~ print zname2, pos, length

        # reading raw data
        if not lazy:
            f.seek(Data_Start_Offset,0)
            rawdata = np.fromstring(f.read() , dtype = 'u'+str(Bytes))
            rawdata = rawdata.reshape(( rawdata.size/Num_Chan , Num_Chan))

        # Reading Code Info
        zname2, pos, length = zones['ORDER']
        f.seek(pos,0)
        code = np.fromfile(f, dtype='u2', count=Num_Chan)

        units = {-1: pq.nano*pq.V, 0:pq.uV, 1:pq.mV, 2:1, 100: pq.percent,  101:pq.dimensionless, 102:pq.dimensionless}

        for c in range(Num_Chan):
            zname2, pos, length = zones['LABCOD']
            f.seek(pos+code[c]*128+2,0)

            label = f.read(6).strip("\x00")
            ground = f.read(6).strip("\x00")
            logical_min , logical_max, logical_ground, physical_min, physical_max = f.read_f('iiiii')
            k, = f.read_f('h')
            if k in units.keys() :
                unit = units[k]
            else :
                unit = pq.uV

            f.seek(8,1)
            sampling_rate, = f.read_f('H') * pq.Hz
            sampling_rate *= Rate_Min

            if lazy:
                signal = [ ]*unit
            else:
                factor = float(physical_max - physical_min) / float(logical_max-logical_min+1)
                signal = ( rawdata[:,c].astype('f') - logical_ground )* factor*unit

            anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                  name=label, channel_index=c)
            if lazy:
                anaSig.lazy_shape = None
            anaSig.annotate(ground = ground)

            seg.analogsignals.append( anaSig )


        sampling_rate = np.mean([ anaSig.sampling_rate for anaSig in seg.analogsignals ])*pq.Hz

        # Read trigger and notes
        for zname, label_dtype in [ ('TRIGGER', 'u2'), ('NOTE', 'S40') ]:
            zname2, pos, length = zones[zname]
            f.seek(pos,0)
            triggers = np.fromstring(f.read(length) , dtype = [('pos','u4'), ('label', label_dtype)] ,  )
            ea = EventArray(name =zname[0]+zname[1:].lower())
            if not lazy:
                keep = (triggers['pos']>=triggers['pos'][0]) & (triggers['pos']<rawdata.shape[0]) & (triggers['pos']!=0)
                triggers = triggers[keep]
                ea.labels = triggers['label'].astype('S')
                ea.times = (triggers['pos']/sampling_rate).rescale('s')
            else:
                ea.lazy_shape = triggers.size
            seg.eventarrays.append(ea)
        
        # Read Event A and B
        # Not so well  tested
        for zname in ['EVENT A', 'EVENT B']:
            zname2, pos, length = zones[zname]
            f.seek(pos,0)
            epochs = np.fromstring(f.read(length) , 
                            dtype = [('label','u4'),('start','u4'),('stop','u4'),]  )
            ep = EpochArray(name =zname[0]+zname[1:].lower())
            if not lazy:
                keep = (epochs['start']>0) & (epochs['start']<rawdata.shape[0]) & (epochs['stop']<rawdata.shape[0])
                epochs = epochs[keep]
                ep.labels = epochs['label'].astype('S')
                ep.times = (epochs['start']/sampling_rate).rescale('s')
                ep.durations = ((epochs['stop'] - epochs['start'])/sampling_rate).rescale('s')
            else:
                ep.lazy_shape = triggers.size
            seg.epocharrays.append(ep)
        
        
        seg.create_many_to_one_relationship()
        return seg





