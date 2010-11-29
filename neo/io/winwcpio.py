# -*- coding: utf-8 -*-
"""

Class for fake reading/writing data from WinWCP, a software tool written by
John Dempster.

WinWCP is free:
http://spider.science.strath.ac.uk/sipbs/software.htm


Supported : Read

@author : sgarcia

"""


from baseio import BaseIO
#from neo.core import *
from neo.core import *

import struct
from numpy import *


class WinWcpIO(BaseIO):
    """
    Class for reading/writing from a WinWCP file.
    
    **Example**
        #read a file
        io = WinWcpIO(filename = 'myfile.wcp')
        blck = io.read() # read the entire file    
    """
    
    is_readable        = True
    is_writable        = False

    supported_objects  = [Block, Segment , AnalogSignal ]
    readable_objects   = [Block]
    writeable_objects  = []  

    has_header         = False
    is_streameable     = False
    
    read_params        = {
                        Block : [
                                ],
                        }
    
    write_params       = None
    
    name               = 'WinWCP'
    extensions          = [ 'wcp' ]
    
    
    mode = 'file'
    
    
    def __init__(self , filename = None) :
        """
        This class read a WinWCP wcp file.
        
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
        return self.read_block( **kargs)
    
    
    
    def read_block(self ):
        """
        Return a Block.
        
        **Arguments**
            no arguments
        
        
        """
        blck = Block()
        fid = open(self.filename , 'rb')
        
        headertext = fid.read(1024)
        header = {}
        for line in headertext.split('\r\n'):
            if '=' not in line : continue
            #print '#' , line , '#'
            key,val = line.split('=')
            if key in ['NC', 'NR','NBH','NBA','NBD','ADCMAX','NP','NZ', ] :
                val = int(val)
            if key in ['AD', 'DT', ] :
                val = val.replace(',','.')
                val = float(val)
            header[key] = val
        
        #print header
        
        SECTORSIZE = 512
        # loop for record number
        for i in range(header['NR']):
            #print 'record ',i
            offset = 1024 + i*(SECTORSIZE*header['NBD']+1024)
            
            # read analysis zone
            analysisHeader = HeaderReader(fid , AnalysisDescription ).read_f(offset = offset)
            #print analysisHeader
            
            # read data
            NP = (SECTORSIZE*header['NBD'])/2
            NP = NP - NP%header['NC']
            NP = NP/header['NC']
            data = memmap(self.filename , dtype('i2')  , 'r', 
                          #shape = (header['NC'], header['NP']) ,
                          shape = (NP,header['NC'], ) ,
                          offset = offset+header['NBA']*SECTORSIZE)
            
            # create a segment
            seg = Segment()
            blck._segments.append(seg)
            
            for c in range(header['NC']):
                anaSig = AnalogSignal()
                seg._analogsignals.append(anaSig)
                YG = float(header['YG%d'%c].replace(',','.'))
                ADCMAX = header['ADCMAX']
                VMax = analysisHeader['VMax'][c]
                anaSig.signal = data[:,header['YO%d'%c]].astype('f4')*VMax/ADCMAX/YG
                anaSig.sampling_rate = 1./analysisHeader['SamplingInterval']
                anaSig.t_start = analysisHeader['TimeRecorded']
                anaSig.name = header['YN%d'%c]
                anaSig.unit = header['YU%d'%c]
                anaSig.channel = c
        
        fid.close()
        return blck
        



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


