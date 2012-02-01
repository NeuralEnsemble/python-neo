# -*- coding: utf-8 -*-
"""
Class for reading data from WinWCP, a software tool written by
John Dempster.

WinWCP is free:
http://spider.science.strath.ac.uk/sipbs/software.htm

Supported : Read

Author : sgarcia

"""

from .baseio import BaseIO
from ..core import *
from .tools import create_many_to_one_relationship

import numpy as np
from numpy import dtype, zeros, fromstring, empty
import quantities as pq

import sys
import os
import struct

PY3K = (sys.version_info[0] == 3)

class WinWcpIO(BaseIO):
    """
    Class for reading from a WinWCP file.

    Usage:
        >>> from neo import io
        >>> r = io.WinWcpIO( filename = 'File_winwcp_1.wcp')
        >>> bl = r.read_block(lazy = False, cascade = True,)
        >>> print bl.segments   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<neo.core.segment.Segment object at 0x1057bd350>, <neo.core.segment.Segment object at 0x1057bd2d0>,
        ...
        >>> print bl.segments[0].analogsignals
        [<AnalogSignal(array([-2438.73388672, -2428.96801758, -2425.61083984, ..., -2695.39453125,
        ...

    """
    
    is_readable        = True
    is_writable        = False

    supported_objects  = [Block, Segment , AnalogSignal ]
    readable_objects   = [Block]
    writeable_objects  = []  

    has_header         = False
    is_streameable     = False
    
    read_params        = { Block : [ ], }
    
    write_params       = None
    
    name               = 'WinWCP'
    extensions          = [ 'wcp' ]
    mode = 'file'
        
    def __init__(self , filename = None) :
        """
        This class read a WinWCP wcp file.
        
        Arguments:
            filename : the filename to read
        
        """
        BaseIO.__init__(self)
        self.filename = filename
    
    def read_block(self , lazy = False,
                                    cascade = True,
                                    ):
        bl = Block( file_origin = os.path.basename(self.filename), )
        if not cascade:
            return bl

        fid = open(self.filename , 'rb')
        
        headertext = fid.read(1024)
        if PY3K:
            headertext = headertext.decode('ascii')
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
            if not lazy:
                data = np.memmap(self.filename , dtype('i2')  , 'r', 
                              #shape = (header['NC'], header['NP']) ,
                              shape = (NP,header['NC'], ) ,
                              offset = offset+header['NBA']*SECTORSIZE)
            
            # create a segment
            seg = Segment()
            bl.segments.append(seg)
            
            for c in range(header['NC']):

                unit = header['YU%d'%c]
                try :
                    unit = pq.Quantity(1., unit)
                except:
                    unit = pq.Quantity(1., '')

                if lazy:
                    signal = [ ] * unit
                else:
                    YG = float(header['YG%d'%c].replace(',','.'))
                    ADCMAX = header['ADCMAX']
                    VMax = analysisHeader['VMax'][c]
                    signal = data[:,header['YO%d'%c]].astype('f4')*VMax/ADCMAX/YG * unit
                anaSig = AnalogSignal(signal ,
                                                    sampling_rate = pq.Hz/analysisHeader['SamplingInterval'] ,
                                                    t_start = analysisHeader['TimeRecorded'] * pq.s,
                                                    name = header['YN%d'%c],
                                                    
                                                        )
                anaSig.annotate(channel_index = c)
                if lazy:
                    anaSig.lazy_shape = NP
                seg.analogsignals.append(anaSig)
        
        fid.close()
        
        create_many_to_one_relationship(bl)
        return bl
        



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


