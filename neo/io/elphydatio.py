# -*- coding: utf-8 -*-
"""
elphyio
==================

Classe for reading/writing data from Elphy

Elphy is a software for acquisition/computing neural data written at Unic in Orsay.


Classes
-------

ElphyDatIO          - Classe for reading/writing data for Elphy file.

@author : sgarcia

"""

import struct
from baseio import BaseIO
from neo.core import *
from numpy import *


class struct_file(file):
    def read_f(self, format , offset = None):
        if offset is not None:
            self.seek(offset)
        return struct.unpack(format , self.read(struct.calcsize(format)))

    def write_f(self, format , offset = None , *args ):
        if offset is not None:
            self.seek(offset)
        self.write( struct.pack( format , *args ) )


class ElphyDatIO(BaseIO):
    """
    Classe for reading/writing data from elphy old format (.DAT)
    
    **Example**
        #read a file
        io = ElphyDatIO(filename = 'myfile.DAT')
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
                                        ]
                            }
    write_params       = None
    
    name               = 'ElphyDac'
    extensions         = [ 'DAT' ]
    
    
    def __init__(self , filename = None) :
        """
        This class read a elphy DAT file.
        
        **Arguments**
            filename : the filename to read
        
        """
        BaseIO.__init__(self)
        self.filename = filename


    def read(self , *args, **kargs):
        """
        Read the file.
        Return a neo.Block by default
        See read_block for detail.
        

        """
        return self.read_block( *args , **kargs)
    
    def read_block(self,  ):
        """
        **Arguments**
            no arguments
        """
        
        block = Block()
        fid = struct_file(self.filename,'rb')
        
        #'16p' is pascal string first byte for length of string read struct.doc
        filetype,  = fid.read_f('16p')
        
        print  'filetype' , filetype
        #test if  DAC 2 file
        if filetype !='DAC2/GS/2000' :
            return None
        
        #header
        headersize, = fid.read_f('i' , offset = 16)
        print 'headersize' , headersize
        
        #1rst block
        block_id = fid.read_f('16p' , offset = 20)
        block_size, = fid.read_f('i'  , offset = 36)
        
        channelCount, = fid.read_f('b' , offset = 40)
        nbpt, = fid.read_f('i' )
        tpdata, = fid.read_f('b')
        unitX, = fid.read_f('11p')
        Dxu, x0u = fid.read_f('dd')
        
        unitY = [ ]
        Dyu = [ ]
        y0u = [ ]
        for channel in range(channelCount) :
            unitY.append(fid.read_f('11p')[0])
            Dyu.append(fid.read_f('d')[0])
            y0u.append(fid.read_f('d')[0])
        print unitX
        if (unitX == 's') or (unitX =='sec') :
            f = 1.
        elif unitX == 'ms' :
            f = 0.001
        else :
            f = 1.
        freq = 1./(Dxu*f)
        t_start = -x0u
        
        preseqI , postseqI , continuous, VariablEepLength , WithTags , = fid.read_f('ii???' , offset = 505)
        
        #data
        fid.seek(headersize)
        
        if continuous :
            datas = fromstring(fid.read() , dtype = 'i2' )
            datas = datas.reshape( (1, datas.shape/channelCount , channelCount))
            nbpt = datas.shape[1]
        else :
            datas = empty((0,nbpt , channelCount), dtype = 'i2')
            while True :
                fid.read(preseqI)
                data = fromstring(fid.read(nbpt*channelCount*2) , dtype = 'i2' )
                if data.size != nbpt*channelCount  : break
                data = data.reshape( (1,nbpt , channelCount))
                datas = concatenate((datas,data))
                fid.read(postseqI)
        fid.close()
        
        datas = datas.astype('f')
        datas *= array(Dyu)[newaxis,newaxis,:]
        datas += array(y0u)[newaxis,newaxis,:]
        
        # block creation
        for s in xrange(datas.shape[0]) :
            seg = Segment()
            for a in xrange(channelCount) :
                anaSig = AnalogSignal(signal = datas[s,:,a] ,
                                        freq = freq,
                                        t_start = t_start)
                seg._analogsignals.append( anaSig )
            block._segments.append(seg)
            
        return block
        
    


