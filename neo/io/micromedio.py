# encoding: utf-8
"""
Class for reading/writing data from micromed (.trc).
Inspired by the Matlab code for EEGLAB from Rami K. Niazy.

Supported : Read


Author: sgarcia

"""

from .baseio import BaseIO
from ..core import *
from .tools import create_many_to_one_relationship
import numpy as np
import quantities as pq


import os
import struct
import datetime

# file no longer exists in Python3
try:
    file
except NameError:
    import io
    file = io.BufferedReader


class struct_file(file):
    def read_f(self, format):
        return struct.unpack(format , self.read(struct.calcsize(format)))


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
    
    supported_objects            = [ Segment , AnalogSignal , EventArray ]
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
        name = f.read(20)
        while name[-1] == ' ' :
            if len(name) == 0 :break
            name = name[:-1]

        #Date
        f.seek(128,0)
        day, month, year = f.read_f('bbb')
        rec_date = datetime.date(year+1900 , month , day)
        
        #header
        f.seek(175,0)
        header_version, = f.read_f('b')
        assert header_version == 4
        
        f.seek(138,0)
        Data_Start_Offset , Num_Chan , Multiplexer , Rate_Min , Bytes = f.read_f('IHHHH')
        f.seek(176+8,0)
        Code_Area , Code_Area_Length, = f.read_f('II')
        f.seek(192+8,0)
        Electrode_Area , Electrode_Area_Length = f.read_f('II')
        f.seek(400+8,0)
        Trigger_Area , Tigger_Area_Length=f.read_f('II')
        
        seg = Segment(  name = name,
                                    file_origin = os.path.basename(self.filename),
                                    )
        seg.annotate(surname = surname)
        seg.annotate(rec_date = rec_date)
        
        if not cascade:
            return seg

        
        # reading raw data
        if not lazy:
            f.seek(Data_Start_Offset,0)
            rawdata = np.fromstring(f.read() , dtype = 'u'+str(Bytes))
            rawdata = rawdata.reshape(( rawdata.size/Num_Chan , Num_Chan))
        
        # Reading Code Info
        f.seek(Code_Area,0)
        code = np.fromfile(f, dtype='u2', count=Num_Chan)
        
        units = {-1: pq.nano*pq.V, 0:pq.uV, 1:pq.mV, 2:1, 100: pq.percent,  101:pq.dimensionless, 102:pq.dimensionless}
        
        for c in range(Num_Chan):
            f.seek(Electrode_Area+code[c]*128+2,0)
            
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

            anaSig = AnalogSignal( signal , sampling_rate = sampling_rate ,name = label)
            if lazy:
                #TODO
                anaSig.lazy_shape = None
            anaSig.annotate(channel_index = c)
            anaSig.annotate(ground = ground)
            
            seg.analogsignals.append( anaSig )
        
        
        sampling_rate = np.mean([ anaSig.sampling_rate for anaSig in seg.analogsignals ])*pq.Hz
        
        # Read trigger
        f.seek(Trigger_Area,0)
        ea = EventArray()
        if not lazy:
            labels = [ ]
            times = [ ]
            first_trig = 0
            for i in range(0,Tigger_Area_Length/6) :
                pos , label = f.read_f('IH')
                if i == 0:
                    first_trig = pos
                if ( pos > first_trig ) and (pos < rawdata.shape[0]) :
                    labels.append(str(label))
                    times.append(pos/sampling_rate)
            ea.labels = np.array(labels)
            ea.times = times*pq.s
        else:
            ea.lazy_shape = Tigger_Area_Length/6
        seg.eventarrays.append(ea)
        
        create_many_to_one_relationship(seg)
        return seg
        
    
    


