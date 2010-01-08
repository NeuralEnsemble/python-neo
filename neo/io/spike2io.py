# -*- coding: utf-8 -*-
"""
exampleio
==================

Classe for fake reading/writing data in a no file.
It is just a example for guidelines for developers who want to develop a new IO.


If you start a new IO class copy/paste and modify.

If you have a problem just mail me or ask the list.


Classes
-------

ExampleIO          - Classe for fake reading/writing data in a no file.


@author : sgarcia

"""


from baseio import BaseIO
from neo.core import *

from numpy import *
from copy import deepcopy

class Spike2IO(BaseIO):
    """
    Class for reading data in smr spike2 CED file.
    
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
                        Segment : [
                                    ],
                        }
    write_params       = None
    level              = None
    nfiles             = 0
    
    name               = 'Spike 2 CED'
    extensions          = [ 'smr' ]
    objects            = []
    supported_types    = [ Block ]
    
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
    
    def read_block(self , filename = '', ):
        """
        Return a fake Block.
        
        **Arguments**
        filename : The filename does not matter.
        
        """
        header = self.read_header(filename = filename)
        fid = open(filename, 'rb')
        for i in range(header.channels) :
            self.readOneChannel( fid, i, header ,)
        
        blck = Block()
        
        
        return blck
        
        
    def read_header(self , filename = ''):
        
        fid = open(filename, 'rb')
        header = HeaderReader(fid,   dtype(headerDescription))
        
        if header.system_id < 6:
            header.dtime_base = 1e-6
            header.datetime_detail = 0
            header.datetime_year = 0
        
        channelHeaders = [ ]
        for i in range(header.channels):
            # read global channel header
            fid.seek(512 + 140*i) # TODO verifier i ou i-1
            channelHeader = HeaderReader(fid, dtype(channelHeaderDesciption1))
            if channelHeader.kind in [1, 6]:
                dt = [('scale' , 'f4'),
                      ('offset' , 'f4'),
                      ('unit' , 'S5'),]
                channelHeader += HeaderReader(fid, dtype(dt))
                if header.system_id < 6:
                    channelHeader += HeaderReader(fid, dtype([ ('divide' , 'i4')]) )#i8
                else : 
                    channelHeader +=HeaderReader(fid, dtype([ ('interleave' , 'i4')]) )#i8
            if channelHeader.kind in [7, 9]:
                dt = [('min' , 'f4'),
                      ('max' , 'f4'),
                      ('unit' , 'S5'),]
                channelHeader += HeaderReader(fid, dtype(dt))
                if header.system_id < 6:
                    channelHeader += HeaderReader(fid, dtype([ ('divide' , 'i4')]))#i8
                else :
                    channelHeader += HeaderReader(fid, dtype([ ('interleave' , 'i4')]) )#i8
            if channelHeader.kind in [4]:
                dt = [('init_low' , 'u1'),
                      ('next_low' , 'u1'),]
                channelHeader += HeaderReader(fid, dtype(dt))
            
            channelHeaders.append(channelHeader)
        
        header.channelHeaders = channelHeaders
        
        fid.close()
        return header

            
    def readOneChannel(self , fid, channel_num, header ,):
        """
        """
        channelHeader = header.channelHeaders[channel_num]
        
        if channelHeader.kind in [1, 9]:
            # read AnalogSignal
            print 'analogChanel'
            # data type
            if channelHeader.kind == 1:
                dt = dtype('i2')
            elif channelHeader.kind == 9:
                dt = dtype('f4')
            
            # read blocks header
            fid.seek(channelHeader.firstblock)
            anaSig = AnalogSignal()
            anaSig.signal = array([ ] , dtype = 'f')
            anaSigs = [ ]
            for b in range(channelHeader.blocks) :
                blockHeader = HeaderReader(fid, dtype(blockHeaderDesciption))
                print blockHeader
                # read data
                sig = fromstring( fid.read(blockHeader.items*dt.itemsize) , dtype = dt)
                anaSig.signal = concatenate( ( anaSig.signal , sig ))
                
                # jump to next block
                if blockHeader.succ_block > 0 :
                    fid.seek(blockHeader.succ_block)
                    nextBlockHeader = HeaderReader(fid, dtype(blockHeaderDesciption))
                    
                    # check is there a continuity with next block
                    sample_interval = (blockHeader.end_time-blockHeader.start_time)/blockHeader.items
                                        
                    interval_with_next = nextBlockHeader.start_time - blockHeader.end_time
                    if interval_with_next > sample_interval:
                        # discontinuous :
                        # create a new anaSig
                        print 'rupture' , sample_interval , interval_with_next
                        anaSigs.append(anaSig)
                        anaSig = AnalogSignal()
                        anaSig.signal = array([ ] , dtype = 'f')
                    
                    fid.seek(blockHeader.succ_block)
            # last one
            anaSigs.append(anaSig)
            print anaSigs
            
            # TODO gerer le dtype si i2
            # TODO gerer heure et freq
            
        elif channelHeader.kind in  [2, 3, 4, 5, 6, 7, 8]:
            # read Event
            pass
            
            
            



class HeaderReader(object):
    def __init__(self , fid , dtype):
        array = fromstring( fid.read(dtype.itemsize) , dtype)[0]
        object.__setattr__(self, 'array' , array)
        object.__setattr__(self, 'dtype' , dtype)
        
    def __setattr__(self, name , val):
        if name in self.dtype.names :
            self.array[name] = val
        else :
            object.__setattr__(self, name , val)

    def __getattr__(self , name):
        if name in self.dtype.names :
            return self.array[name]
        else :
            object.__getattr__(self, name )
    def names(self):
        return self.array.dtype.names
    
    def __repr__(self):
        print 'HEADER'
        for name in self.dtype.names :
            if self.dtype[name].kind != 'S' :
                print name , self.array[name]
        print ''
        return ''
    
    def __add__(self, header2):
        #print 'add' , self.dtype, header2.dtype
        newdtype = [ ]
        for name in self.dtype.names :
            newdtype.append( (name , self.dtype[name].str) )
        for name in header2.dtype.names :
            newdtype.append( (name , header2.dtype[name].str) )
        newdtype = dtype(newdtype)
        
        newHeader = deepcopy(self)
        newHeader.dtype = newdtype
        newHeader.array = fromstring( self.array.tostring()+header2.array.tostring() , newdtype)[0]
        return newHeader

# headers structures :
headerDescription = [
    ( 'system_id', 'i2' ),
    ( 'copyright', 'S10' ),
    ( 'creator', 'S8' ),
    ( 'us_per_time', 'i2' ),
    ( 'time_per_adc', 'i2' ),
    ( 'filestate', 'i2' ),
    ( 'first_data', 'i4' ),#i8
    ( 'channels', 'i2' ),
    ( 'chan_size', 'i2' ),
    ( 'extra_data', 'i2' ),
    ( 'buffersize', 'i2' ),
    ( 'os_format', 'i2' ),
    ( 'max_ftime', 'i4' ),#i8
    ( 'dtime_base', 'f8' ),
    ( 'datetime_detail', 'u1' ),
    ( 'datetime_year', 'i2' ),
    ( 'pad', 'S52' ),
    ( 'comment1', 'S80' ),
    ( 'comment2', 'S80' ),
    ( 'comment3', 'S80' ),
    ( 'comment4', 'S80' ),
    ( 'comment5', 'S80' ),
    ]

channelHeaderDesciption1 = [
    ('del_size','i2'),
    ('next_del_block','i4'),#i8
    ('firstblock','i4'),#i8
    ('lastblock','i4'),#i8
    ('blocks','i2'),
    ('n_extra','i2'),
    ('pre_trig','i2'),
    ('free0','i2'),
    ('py_sz','i2'),
    ('max_data','i2'),
    ('comment','S72'),
    ('max_chan_time','i4'),#i4
    ('l_chan_dvd','i4'),#i4
    ('phy_chan','i2'),
    ('title','S10'),
    ('ideal_rate','f4'),
    ('kind','u1'),
    ('unused1','i1'),
    
    ]

blockHeaderDesciption =[
    ('pred_block','i4'),#i4
    ('succ_block','i4'),#i4
    ('start_time','i4'),#i4
    ('end_time','i4'),#i4
    ('channel_num','i2'),
    ('items','i2'),
    ]
