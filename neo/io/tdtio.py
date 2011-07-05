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
        if self.dirname.endswith('/'):
            self.dirname = self.dirname[:-1]
        tankname = os.path.basename(self.dirname)
        for blockname in os.listdir(self.dirname):
            subdir = os.path.join(self.dirname,blockname)
            if not os.path.isdir(subdir): continue
            
            seg = Segment()
            blck._segments.append( seg)
            
            
            global_t_start = None
            # Step 1 : first loop for counting - tsq file
            tsq = open(os.path.join(subdir, tankname+'_'+blockname+'.tsq'), 'rb')
            hr = HeaderReader(tsq, TsqDescription)
            allsig = { }
            allspiketr = { }
            while 1:
                h= hr.read_f()
                if h==None:break
                
                
                if Types[h['type']] == 'EVTYPE_UNKNOWN':
                    pass
                    
                elif Types[h['type']] == 'EVTYPE_MARK' :
                    # This event marks the beginning of the recording session
                    # in Unix time (seconds since 1970). Spike times
                    # are stored in the same format, so we will subtract
                    # global_t_start from them.
                    if global_t_start is None:
                        global_t_start = h['timestamp']
                
                elif Types[h['type']] == 'EVTYPE_SCALER' :
                    # TODO
                    pass
                
                elif Types[h['type']] == 'EVTYPE_STRON' or \
                     Types[h['type']] == 'EVTYPE_STROFF':
                     
                    ev = Event()
                    ev.time = h['timestamp'] - global_t_start
                    ev.name = h['code']
                     # it the strobe attribute masked with eventoffset
                    strobe, = struct.unpack('d' , struct.pack('q' , h['eventoffset']))
                    ev.label = str(strobe)
                    seg._events.append( ev )

                elif Types[h['type']] == 'EVTYPE_SNIP' :
                    # TODO
                    if h['code'] not in allspiketr:
                        allspiketr[h['code']] = { }
                    if h['channel'] not in allspiketr[h['code']]:
                        allspiketr[h['code']][h['channel']] = { }
                    if h['sortcode'] not in allspiketr[h['code']][h['channel']]:
                        sptr = SpikeTrain()
                        sptr.channel = h['channel']
                        sptr.name = str(h['sortcode'])
                        # We start the SpikeTrain at time zero
                        # But is this right, or should we get the start
                        # time of the AnalogSignal?
                        sptr.t_start = 0. #global_t_start
                        sptr.sampling_rate = h['frequency']
                        sptr.left_sweep = (h['size']-10.)/2./h['frequency']
                        sptr.right_sweep = (h['size']-10.)/2./h['frequency']
                        sptr.nbspike = 0
                        sptr.pos = 0
                        sptr.waveformsize = h['size']-10
                        
                        allspiketr[h['code']][h['channel']][h['sortcode']] = sptr
                    
                    allspiketr[h['code']][h['channel']][h['sortcode']].nbspike += 1
                
                elif Types[h['type']] == 'EVTYPE_STREAM':
                    if h['code'] not in allsig:
                        allsig[h['code']] = { }
                    if h['channel'] not in allsig[h['code']]:
                        anaSig = AnalogSignal( 
                                                            channel = h['channel'],
                                                            name =  h['code'],
                                                            signal = None,
                                                            sampling_rate = h['frequency'],
                                                            t_start = h['timestamp'] - global_t_start,
                                                            )
                        anaSig.dtype =  dtype(DataFormats[h['dataformat']])
                        anaSig.totalsize = 0
                        anaSig.pos = 0
                        allsig[h['code']][h['channel']] = anaSig
                    #allsig[h['code']][h['channel']].totalsize += (h['size']*4-40)/anaSig.dtype.itemsize
                    allsig[h['code']][h['channel']].totalsize += \
                        (h['size']*4-40)/allsig[h['code']][h['channel']].dtype.itemsize
            
            # Step 2 : allocate memory
            for code, v in allsig.iteritems():
                for channel, anaSig in v.iteritems():
                    anaSig.signal = zeros( anaSig.totalsize , dtype = anaSig.dtype )
                    pass
            
            for code, v in allspiketr.iteritems():
                for channel, allsorted in v.iteritems():
                    for sortcode, sptr in allsorted.iteritems():
                        sptr._spike_times = zeros( (sptr.nbspike), dtype ='f')
                        sptr._waveforms = zeros( (sptr.nbspike, 1 , sptr.waveformsize), dtype = 'f')
            
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
            for code, v in allspiketr.iteritems():
                for channel, allsorted in v.iteritems():
                    for sortcode, sptr in allsorted.iteritems():
                        sptr.fid = tev



            # Step 4 : second loop for copyin chunk of data
            tsq.seek(0)
            while 1:
                h= hr.read_f()
                if h==None:break
                    
                if Types[h['type']] == 'EVTYPE_STREAM': 
                    # The signal to put the data into
                    a = allsig[h['code']][h['channel']]
                    dt = a.dtype
                    
                    # The number of samples
                    s = (h['size']*4-40)/dt.itemsize
                    
                    # Read the data from position given in h from file in a
                    a.fid.seek(h['eventoffset'])
                    
                    # We read `number of samples` * `sample width` bytes
                    # And store it in signal `a`
                    a.signal[ a.pos:a.pos+s ]  = fromstring( a.fid.read( s*dt.itemsize ), dtype = a.dtype)
                    
                    # Update our position in signal `a`
                    a.pos += s
                    
                elif Types[h['type']] == 'EVTYPE_SNIP': 
                    # Get the spiketrain for this neuron
                    sptr = allspiketr[h['code']][h['channel']][h['sortcode']]
                    
                    # The spike time is the timestamp (Unix time) minus
                    # the start of the session.
                    sptr._spike_times[sptr.pos] = h['timestamp'] - global_t_start
                    
                    # Read the spike waveform from the next chunk of data
                    # in the format specified in the header
                    if h['dataformat'] == 0:
                        # 32 bit
                        sptr._waveforms[sptr.pos, 0, :] = \
                            fromstring( sptr.fid.read( \
                            sptr.waveformsize*4 ), 
                            dtype = DataFormats[h['dataformat']])
                    elif h['dataformat'] == 4:
                        # 64 bit
                        sptr._waveforms[sptr.pos, 0, :] = \
                            fromstring( sptr.fid.read( \
                            sptr.waveformsize*8 ), 
                            dtype = DataFormats[h['dataformat']])
                    sptr.pos += 1
                
            
            # Step 5 : populating segment
            
            for code, v in allsig.iteritems():
                for channel, anaSig in v.iteritems():
                    del anaSig.totalsize
                    del anaSig.pos
                    del anaSig.fid
                    seg._analogsignals.append( anaSig )

            for code, v in allspiketr.iteritems():
                for channel, allsorted in v.iteritems():
                    for sortcode, sptr in allsorted.iteritems():
                        del sptr.nbspike
                        del sptr.waveformsize
                        del sptr.pos
                        del sptr.fid
                        seg._spiketrains.append( sptr )
                    


            
        
        return blck


# From the tank format specification, the size of each of the data
# in each header
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

# Types of events
Types =    {
                0x0 : 'EVTYPE_UNKNOWN',
                0x101:'EVTYPE_STRON',
                0x102:'EVTYPE_STROFF',
                0x201:'EVTYPE_SCALER',
                0x8101:'EVTYPE_STREAM',
                0x8201:'EVTYPE_SNIP',
                0x8801: 'EVTYPE_MARK',
                }

# Keys for each data format
DataFormats = {
                        0 : float32,
                        1 : int32,
                        2 : int16,
                        3 : int8,
                        4 : float64,
                        #~ 5 : ''
                        }




# Reads the header and returns a dict with the values in it,
# one value for each of the entries in TsqDescription
class HeaderReader():
    def __init__(self,fid ,description ):
        # description is just TsqDescription from above
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




