# -*- coding: utf-8 -*-
"""

Classe for reading/writing data from micromed (.trc).

Inspired from the Matlab code for EEGLAB from Rami K. Niazy.

Supported : Read

@author : sgarcia

"""

import struct
from baseio import BaseIO
#from neo.core import *
from ..core import *
from numpy import *
import re
import datetime


class struct_file(file):
    def read_f(self, format):
        return struct.unpack(format , self.read(struct.calcsize(format)))


class MicromedIO(BaseIO):
    """
    Classe for reading/writing data from micromed (.trc).
    
    **Usage**

    **Example**
        #read a file
        io = MicromedIO(filename = 'myfile.TRC')
        seg = io.read() # read the entire file    
    """
    
    is_readable        = True
    is_writable        = False
    
    supported_objects            = [ Segment , AnalogSignal , Event]
    readable_objects    = [Segment]
    writeable_objects    = []      
    
    has_header         = False
    is_streameable     = False
    read_params        = {
                        Segment : [
                                    ('averagesamplerate' , {'value' : True, 'label' : 'if samplerate is inexact for signal then average'  }) ,
                                    ]
                        }
    write_params       = None
    
    name               = None
    extensions          = [ 'TRC' ]
    
    def __init__(self , filename = None) :
        """
        This class read a micromed TRC file.
        
        **Arguments**
            filename : the filename to read
        
        """
        BaseIO.__init__(self)
        self.filename = filename


    def read(self , **kargs):
        """
        Read the file.
        Return a neo.Segment
        See read_segment for detail.
        """
        return self.read_segment( **kargs)
    
    def read_segment(self, averagesamplerate = True,):
        """
        **Arguments**
            averagesamplerate : don't known exactly because this code is reverse ingeneering from another code
                        I supposed that in some case all samplerate are not exaclty the same due to rounding problem
                        in that case average all sample rate
        """
        
        seg = Segment()
        
        f = struct_file(self.filename, 'rb')
        
        #-- Name
        f.seek(64,0)
        surname = f.read(22)
        while surname[-1] == ' ' : 
            if len(surname) == 0 :break
            surname = surname[:-1]
        name = f.read(20)
        while name[-1] == ' ' :
            if len(name) == 0 :break
            name = name[:-1]
        #~ seg.name = name
        #~ seg.surname = surname


        #-- Date
        f.seek(128,0)
        day, month, year = f.read_f('bbb')
        thedate = datetime.date(year+1900 , month , day)
        seg.date = thedate
        
        #header
        f.seek(175,0)
        header_version, = f.read_f('b')
        if header_version!=4 :
            raise('*.trc file is not Micromed System98 Header type 4')
        
        f.seek(138,0)
        Data_Start_Offset , Num_Chan , Multiplexer , Rate_Min , Bytes = f.read_f('IHHHH')
        print 'Data_Start_Offset , Num_Chan , Multiplexer , Rate_Min , Bytes' , Data_Start_Offset , Num_Chan , Multiplexer , Rate_Min , Bytes
        f.seek(176+8,0)
        Code_Area , Code_Area_Length, = f.read_f('II')
        f.seek(192+8,0)
        Electrode_Area , Electrode_Area_Length = f.read_f('II')
        print 'Electrode_Area , Electrode_Area_Length' , Electrode_Area , Electrode_Area_Length
        f.seek(400+8,0)
        Trigger_Area , Tigger_Area_Length=f.read_f('II')
        print 'Trigger_Area , Tigger_Area_Length' , Trigger_Area , Tigger_Area_Length
        
        # reading raw data
        f.seek(Data_Start_Offset,0)
        rawdata = fromstring(f.read() , dtype = 'u'+str(Bytes))
        rawdata = rawdata.reshape(( rawdata.size/Num_Chan , Num_Chan))
        print rawdata.shape
        # Reading Code Info
        f.seek(Code_Area,0)
        code = fromfile(f, dtype='u2', count=Num_Chan)
        
        units = {-1:1e-9, 0:1e-6, 1:1e-3, 2:1, 100:'percent', 101:'bpm', 102:'Adim'}
        for c in range(Num_Chan):
            f.seek(Electrode_Area+code[c]*128+2,0)
            
            #-- inputs
            #positive_input = "%02d-%s" % (c,f.read(6).strip("\x00"))
            #negative_input = "%02d-%s" % (c,f.read(6).strip("\x00"))
            label = f.read(6).strip("\x00")
            ground = f.read(6).strip("\x00")
            
            #-- min and max
            logical_min , logical_max, logical_ground, physical_min, physical_max = f.read_f('iiiii')
            #-- unit
            k, = f.read_f('h')
            if k in units.keys() :
                unit = units[k]
            else :
                unit = 10e-6
            #-- rate
            f.seek(8,1)
            freq, = f.read_f('H')
            #print 'freq' , freq,
            freq *= Rate_Min
            #print freq
            # signal
            factor = float(physical_max - physical_min) / float(logical_max-logical_min+1)
            if type(unit) != str :
                factor *= unit
            signal = ( rawdata[:,c].astype('f') - logical_ground )* factor
            anaSig = AnalogSignal(freq = freq , signal = signal, t_start =0.)
            anaSig.label = label
            anaSig.ground = ground
            anaSig.num = c
            seg._analogsignals.append( anaSig )
            
        freq = mean([ anaSig.freq for anaSig in seg._analogsignals ])
        if averagesamplerate :
            for anaSig in seg._analogsignals :
                anaSig.freq = freq
        
        # Read trigger
        if True :
            f.seek(Trigger_Area,0)
            first_trig = 0
            for i in range(0,Tigger_Area_Length/6) :
                pos , label = f.read_f('IH')
                #print pos, label
                if ( i == 0 )  :
                    first_trig = pos
                if ( pos >= first_trig ) and (pos <= rawdata.shape[0]) :
                    ev = Event( time = pos/freq )
                    ev.label = label
                    seg._events.append( ev )
        
        return seg
        
    

        
