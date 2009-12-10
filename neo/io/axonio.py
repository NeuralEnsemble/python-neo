# -*- coding: utf-8 -*-
"""
axonio
==================

Classe for reading/writing data from pCLAMP and AxoScope 
files (.abf version 1 and 2), develloped by Molecular device/Axon technologies.

abf = Axon binary file

atf is a text file based from axon that can be read by AsciiIO.
but this file is less efficient.

This code is a port abfload and abf2load
written in Matlab by
Copyright (c) 2009, Forrest Collman 
                    fcollman@princeton.edu
Copyright (c) 2004, Harald Hentschke
and disponible here :
http://www.mathworks.com/matlabcentral/fileexchange/22114-abf2load

information on abf 1 and 2 format are disponible here:
http://www.moleculardevices.com/pages/software/developer_info.html


Classes
-------

AxonIO          - Classe for reading/writing data in abf axon files.

@author : sgarcia

"""

import struct
from baseio import BaseIO
from neo.core import *
from numpy import *
import re
import datetime


class struct_file(file):
    def read_f(self, format):
        return struct.unpack(format , self.read(struct.calcsize(format)))
        
    def write_f(self, format , offset = None , *args ):
        if offset is not None:
            self.seek(offset)
        self.write( struct.pack( format , *args ) )



class AxonIO(BaseIO):
    """
    Classe for reading/writing data from axon binary file(.abf)
    
    **Usage**

    **Example**
    
    """
    
    is_readable        = True
    is_writable        = True
    is_object_readable = False
    is_object_writable = False
    has_header         = False
    is_streameable     = False
    read_params        = {}
    write_params       = {}
    level              = None
    nfiles             = 0
    name               = None
    objects            = []
    supported_types    = []
    
    def __init__(self ) :
        """
        
        **Arguments**
        
        """
        
        BaseIO.__init__(self)


    def read(self , *args, **kargs):
        """
        Read the file.
        Return a neo.Block by default
        See read_block for detail.
        
        You can also call read_segment if you assume that your file contain only
        one Segment.
        """
        return self.read_block( *args , **kargs)
    
    def read_block(self, filename = '', ):
        """
        **Arguments**
            filename : filename
            TODO
        """
        
        block = Block()
        fid = struct_file(filename,'rb')
        print fid.read(4)
        
        
        fid.close()
        return block
