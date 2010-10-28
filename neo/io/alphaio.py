# -*- coding: utf-8 -*-
"""

Class for reading data from AlphaOmega system files (map)

This code is written from file specificaion downloaded here:
TODO

Supported : Read


@author : sgarcia, Florent Jaillet

"""

from baseio import BaseIO
from ..core import *

import datetime

#~ from numpy import *
import struct


class AlphaIO(BaseIO):
    """
    
    """
    
    is_readable        = True
    is_writable        = False
    supported_objects  = [ Block, Segment , AnalogSignal, SpikeTrain, Event, Epoch]
    readable_objects    = [ Block] 
    writeable_objects   = []

    has_header         = False
    is_streameable     = False
    
    read_params        = {
                        Block : [
                                    ],
                        }
    
    write_params       = None
    
    name               = 'AlphaOmega'
    extensions          = [ 'map' ]
    mode = 'file' 
    

    
    def __init__(self , filename = None) :
        """
        This class read a abf file.
        
        **Arguments**
        
            filename : the filename to read you can pu what ever it do not read anythings
        
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

    
    def read_block(self):
        bl = Block()

        fid = open(self.filename, 'rb')
        globalHeader = HeaderReader(fid , TypeH_Header ).read_f(offset = 0)
        print globalHeader
        
        
        m_length, m_TypeBlock = struct.unpack('hs' , fid.read(3))
        print m_length, m_TypeBlock
        
        return bl
    



TypeH_Header = [
                ('m_nextBlock' , 'i'),
                ('m_version','h'),
                ('m_time','8s'),
                ('m_date','8s'),
                ('m_MinimumTime','f'),
                ('m_MaximumTime','f'),
                
            ]


dict_header_type = {
                                'h' : TypeH_Header,
                                #~ '1' : 
                                }

    
    
    
class HeaderReader():
    def __init__(self,fid ,description ):
        self.fid = fid
        self.description = description
    def read_f(self, offset =None):
        if offset is not None :
            self.fid.seek(offset)
        d = { }
        for key, format in self.description :
            buf = self.fid.read(struct.calcsize(format))
            if len(buf) != struct.calcsize(format) : return None
            val = struct.unpack(format , buf)
            if len(val) == 1:
                val = val[0]
            else :
                val = list(val)
            if 's' in format :
                val = val.replace('\x00','')
            d[key] = val
        return d

