# -*- coding: utf-8 -*-
"""

Class for reading data from brain vision system files (vhdr)

This code is written from the file specification downloaded here:
TODO

Supported : Read


@author : Simon More, sgarcia

"""

from baseio import BaseIO
from neo.core import *

import datetime

#~ from numpy import *
import struct


class BrainVisionIO(BaseIO):
    """
    
    """
    
    is_readable        = True
    is_writable        = False
    supported_objects  = [ Block, Segment , AnalogSignal, Event, Epoch ]
    readable_objects    = [ Block] 
    writeable_objects   = []

    has_header         = False
    is_streameable     = False
    
    read_params        = {
                        Block : [
                                    ],
                        }
    
    write_params       = None
    
    name               = 'BrainVision'
    extensions          = [ 'vhdr' ]
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
        
        return bl
    


