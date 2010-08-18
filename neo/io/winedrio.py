# -*- coding: utf-8 -*-
"""

Classe for fake reading/writing data from WinEdr a software written written by
John Dempster.

WinEdr is free:
http://spider.science.strath.ac.uk/sipbs/software.htm


Supported : Read

@author : sgarcia

"""


from baseio import BaseIO
#from neo.core import *
from ..core import *

import struct
from numpy import *


class WinEdrIO(BaseIO):
    """
    Class for reading/writing from a WinEDR file.
    
    **Example**
        #read a file
        io = WinEdrIO(filename = 'myfile.EDR')
        blck = io.read() # read the entire file    
    """
    
    is_readable        = True
    is_writable        = False

    supported_objects  = [ Segment , AnalogSignal ]
    readable_objects   = [Segment]
    writeable_objects  = []  

    has_header         = False
    is_streameable     = False
    
    read_params        = {
                        Segment : [
                                ],
                        }
    
    write_params       = None
    
    name               = 'WinEDR'
    extensions          = [ 'EDR' ]
    
    
    mode = 'file'
    
    def __init__(self , filename = None) :
        """
        This class read a WinEDR wcp file.
        
        **Arguments**
            filename : the filename to read
        
        """
        BaseIO.__init__(self)
        self.filename = filename


    def read(self , **kargs):
        """
        Read a fake file.
        Return a neo.Block
        See read_block for detail.
        """
        return self.read_segment( **kargs)
    
    
    
    def read_segment(self ):
        """
        Return a Block.
        
        **Arguments**
            no arguments
        
        
        """
        seg = Segment()
        fid = open(self.filename , 'rb')
        
        headertext = fid.read(2048)
        header = {}
        for line in headertext.split('\r\n'):
            if '=' not in line : continue
            #print '#' , line , '#'
            key,val = line.split('=')
            if key in ['NC', 'NR','NBH','NBA','NBD','ADCMAX','NP','NZ','ADCMAX' ] :
                val = int(val)
            if key in ['AD', 'DT', ] :
                val = val.replace(',','.')
                val = float(val)
            header[key] = val
        
        data = memmap(self.filename , dtype('i2')  , 'r', 
              #shape = (header['NC'], header['NP']) ,
              shape = (header['NP']/header['NC'],header['NC'], ) ,
              offset = header['NBH'])

        for c in range(header['NC']):
            anaSig = AnalogSignal()
            seg._analogsignals.append(anaSig)
            
            YCF = float(header['YCF%d'%c].replace(',','.'))
            YAG = float(header['YAG%d'%c].replace(',','.'))
            YZ = float(header['YZ%d'%c].replace(',','.'))
            
            ADCMAX = header['ADCMAX']
            AD = header['AD']
            DT = header['DT']
            
            if 'TU' in header:
                if header['TU'] == 'ms':
                    DT *= .001
            
            anaSig.signal = (data[:,header['YO%d'%c]].astype('f4')-YZ) *AD/( YCF*YAG*(ADCMAX+1))
            anaSig.sampling_rate = 1./DT
            anaSig.t_start = 0
            anaSig.name = header['YN%d'%c]
            anaSig.unit = header['YU%d'%c]
            anaSig.channel = c            
            
        return seg
        
        
        



AnalysisDescription = [
    ('RecordStatus','8s'),
    ('RecordType','4s'),
    ('GroupNumber','f'),
    ('TimeRecorded','f'),
    ('SamplingInterval','f'),
    ('VMax','8f'),
    ]


class HeaderReader():
    def __init__(self,fid ,description ):
        self.fid = fid
        self.description = description
    def read_f(self, offset =0):
        self.fid.seek(offset)
        d = { }
        for key, format in self.description :
            val = struct.unpack(format , self.fid.read(struct.calcsize(format)))
            if len(val) == 1:
                val = val[0]
            else :
                val = list(val)
            d[key] = val
        return d


