# -*- coding: utf-8 -*-
"""

Class for fake reading/writing data from Tunker Davis TTank format.


Supported : Read

@author : sgarcia

"""


from baseio import BaseIO
#from neo.core import *
from ..core import *

import struct
from numpy import *
import os

class TdtIO(BaseIO):
    """
    Class for fake reading/writing data from Tunker Davis TTank format.
    
    **Example**
        #read a file
        io = TdtIO(filename = 'myfile.EDR')
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
    
    name               = 'TDT'
    extensions          = [ ]
    
    mode = 'dir'
    
    def __init__(self , dirname = None) :
        """
        This class read a WinEDR wcp file.
        
        **Arguments**
            filename : the filename to read
        
        """
        BaseIO.__init__(self)
        self.dirname = dirname


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
        
        tankname = os.path.basename(self.dirname)
        for blockname in os.listdir(self.dirname):
            subdir = os.path.join(self.dirname,blockname)
            if not os.path.isdir(subdir): continue
            
            # Step 1 : first loop for counting - tsq file
            tsq = open(os.path.join(subdir, tankname+'_'+blockname+'.tsq'), 'rb')
            hr = HeaderReader(tsq, TsqDescription)
            allsig = { }
            while 1:
                h= hr.read_f()
                if h==None:break
                
                
                if Types[h['type']] != 'EVTYPE_STREAM':
                    # TODO
                    continue
                
                if h['code'] not in allsig:
                    allsig[h['code']] = { }
                if h['channel'] not in allsig[h['code']]:
                    anaSig = AnalogSignal( 
                                                        channel = h['channel'],
                                                        name =  h['code'],
                                                        signal = None,
                                                        sampling_rate = h['frequency'],
                                                        t_start = h['timestamp'],
                                                        )
                    anaSig.dtype =  dtype(DataFormats[h['dataformat']])
                    anaSig.totalsize = 0
                    anaSig.pos = 0
                    allsig[h['code']][h['channel']] = anaSig
                allsig[h['code']][h['channel']].totalsize += (h['size']*4-40)/anaSig.dtype.itemsize
            
            # Step 2 : allocate memory
            for code, v in allsig.iteritems():
                for channel, anaSig in v.iteritems():
                    anaSig.signal = zeros( anaSig.totalsize , dtype = anaSig.dtype )
            
            # Step 3 : searh sev (individual data files) or tev (common data file)
            # sev is for version > 70
            if os.path.exists(os.path.join(subdir, tankname+'_'+blockname+'.tev')):
                tev = open(os.path.join(subdir, tankname+'_'+blockname+'.tev'), 'rb')
            else:
                tev = None
            for code, v in allsig.iteritems():
                for channel, anaSig in v.iteritems():
                    filename = os.path.join(subdir, tankname+'_'+blockname+'_'+anaSig.name+'_ch'+str(anaSig.channel)+'.sev')
                    if os.path.exists(filename):
                        anaSig.fid = open(filename, 'rb')
                    else:
                        anaSig.fid = tev
            
            # Step 4 : second loop for copyin chunk of data
            tsq.seek(0)
            while 1:
                h= hr.read_f()
                if h==None:break
                
                if Types[h['type']] != 'EVTYPE_STREAM': continue
                
                a = allsig[h['code']][h['channel']]
                dt = a.dtype
                s = (h['size']*4-40)/dt.itemsize
                a.fid.seek(h['eventoffset'])
                a.signal[ a.pos:a.pos+s ]  = fromstring( a.fid.read( s*dt.itemsize ), dtype = a.dtype)
                a.pos += s
            
            # Step 5 : populating segment
            seg = Segment()
            blck._segments.append( seg)
            
            for code, v in allsig.iteritems():
                for channel, anaSig in v.iteritems():
                    del anaSig.totalsize
                    del anaSig.pos
                    del anaSig.fid
                    seg._analogsignals.append( anaSig )
        
        return blck


TsqDescription = [
    ('size','i'),
    ('type','i'),
    ('code','4s'),
    ('channel','H'),
    ('sortcode','H'),
    ('timestamp','d'),
    ('eventoffset','q'),
    ('dataformat','i'),
    ('frequency','f'),
    ]

Types =    {
                0x0 : 'EVTYPE_UNKNOWN',
                0x101:'EVTYPE_STRON',
                0x102:'EVTYPE_STROFF',
                0x201:'EVTYPE_SCALER',
                0x8101:'EVTYPE_STREAM',
                0x8201:'EVTYPE_SNIP',
                0x8801: 'EVTYPE_MARK',
                }
DataFormats = {
                        0 : float32,
                        1 : int32,
                        2 : int16,
                        3 : int8,
                        4 : float64,
                        #~ 5 : ''
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
            #~ if 's' in format :
                #~ val = val.replace('\x00','')
            d[key] = val
        return d




