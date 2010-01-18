# -*- coding: utf-8 -*-
"""
winwcpio
==================

Classe for fake reading/writing data from WinWCP a software written written by
John Dempster.

WinWCP is free:
http://spider.science.strath.ac.uk/sipbs/software.htm



Classes
-------

WinWcpIO          - Classe for fake reading/writing data in a no file.


@author : sgarcia

"""


from baseio import BaseIO
from neo.core import *
import struct
from numpy import *


class WinWcpIO(BaseIO):
    """
    Class for reading/writing from a WinWCP file.
    
    **Usage**

    **Example**
    
    """
    
    is_readable        = True
    is_writable        = False
    is_object_readable = True
    is_object_writable = False
    has_header         = False
    is_streameable     = False
    
    read_params        = {
                        Block : [
                                ],
                        }
    
    write_params       = None
    
    level              = None
    nfiles             = 0
    
    name               = 'WinWCP'
    extensions          = [ 'wcp' ]
    objects            = []
    supported_types    = [ Block]
    
    def __init__(self ) :
        """
        
        **Arguments**
        
        """
        
        BaseIO.__init__(self)


    def read(self , **kargs):
        """
        Read a fake file.
        Return a neo.Block
        See read_block for detail.
        """
        return self.read_block( **kargs)
    
    
    
    def read_block(self , filename = '',):
        """
        Return a Block.
        
        **Arguments**
        filename : The filename does not matter.
        
        
        """
        blck = Block()
        fid = open(filename , 'rb')
        
        headertext = fid.read(1024)
        print headertext
        header = {}
        for line in headertext.split('\r\n'):
            
            if '=' not in line : continue
            print '#' , line , '#'
            key,val = line.split('=')
            if key in ['NC', 'NR','NBH','NBA','NBD','ADCMAX','NP','NZ', ] :
                val = int(val)
            if key in ['AD', 'DT', ] :
                val = val.replace(',','.')
                val = float(val)
            header[key] = val
        
        print header
        
        SECTORSIZE = 512
        # loop for record number
        for i in range(header['NR']):
            print 'record ',i
            
            # PAS de NP !!!! dans version 8
            #offset = 1024 + i*(2*header['NC']*header['NP']+1024)
            
            #fid.seek(1024 + i*(SECTORSIZE*header['NBD']+1024) - 1024 )
            #yep = fid.read(1024)
            #print yep
            offset = 1024 + i*(SECTORSIZE*header['NBD']+1024)
            
            
            
            #fid.seek(offset)
            # read analysis zone
            analysisHeader = HeaderReader(fid , AnalysisDescription ).read_f(offset = offset)
            print analysisHeader
            
            NP = (SECTORSIZE*header['NBD'])/2
            NP = NP - NP%header['NC']
            NP = NP/header['NC']
            print 'NP', NP
            samplelength = int(floor(analysisHeader['TimeRecorded']/analysisHeader['SamplingInterval']))
            print 'samplelength',samplelength
            
            # read data
            data = memmap(filename , dtype('i2')  , 'r', 
                          #shape = (header['NC'], header['NP']) ,
                          shape = (header['NC'], NP) ,
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
                anaSig.freq = 1./analysisHeader['SamplingInterval']
                anaSig.t_start = 0
                anaSig.name = header['YN%d'%c]
                anaSig.unit = header['YU%d'%c]
        
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


