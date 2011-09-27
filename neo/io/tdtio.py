# encoding: utf-8
"""
Class for reading data from from Tucker Davis TTank format.
Terminology: 
TDT hold data with tanks (actually a directory). And tanks hold sub block (sub directories).
Tanks correspond to neo.Block and tdt block correspond to neo.Segment.

Note the name Block is ambiguous because it does not refer to same thing in TDT terminilogy and neo.


Depend on: 

Supported : Read

Author: sgarcia

"""

from .baseio import BaseIO
from ..core import *
from .tools import create_many_to_one_relationship

import numpy as np
from numpy import dtype
import quantities as pq

import struct
import os

class TdtIO(BaseIO):
    """
    Class for reading data from from Tucker Davis TTank format.
    
    Usage:
        >>> from neo import io
        >>> r = io.TdtIO(dirname='aep_05')
        >>> bl = r.read_block(lazy=False, cascade=True)
        >>> print bl.segments
        [<neo.core.segment.Segment object at 0x1060a4d10>]
        >>> print bl.segments[0].analogsignals
        [<AnalogSignal(array([ 2.18811035,  2.19726562,  2.21252441, ...,  1.33056641,
                1.3458252 ,  1.3671875 ], dtype=float32) * pA, [0.0 s, 191.2832 s], sampling rate: 10000.0 Hz)>]
        >>> print bl.segments[0].eventarrays
        []
    """
    
    
    
    is_readable        = True
    is_writable        = False

    supported_objects  = [Block, Segment , AnalogSignal, EventArray ]
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
        Arguments:
            dirname: path of the TDT tank (a directory)
        
        """
        BaseIO.__init__(self)
        self.dirname = dirname
        if self.dirname.endswith('/'):
            self.dirname = self.dirname[:-1]

    def read_block(self,
                                        lazy = False,
                                        cascade = True,
                                ):
        bl = Block()
        tankname = os.path.basename(self.dirname)
        bl.file_origin = tankname
        if not cascade : return bl
        for blockname in os.listdir(self.dirname):
            if blockname == 'TempBlk': continue
            subdir = os.path.join(self.dirname,blockname)
            
            if not os.path.isdir(subdir): continue
            
            seg = Segment(name = blockname)
            bl.segments.append( seg)
            
            
            global_t_start = None
            # Step 1 : first loop for counting - tsq file
            tsq = open(os.path.join(subdir, tankname+'_'+blockname+'.tsq'), 'rb')
            hr = HeaderReader(tsq, TsqDescription)
            allsig = { }
            allspiketr = { }
            allevent = { }
            while 1:
                h= hr.read_f()
                if h==None:break
                
                channel, code ,  evtype = h['channel'], h['code'], h['evtype']
                
                if Types[evtype] == 'EVTYPE_UNKNOWN':
                    pass
                    
                elif Types[evtype] == 'EVTYPE_MARK' :
                    if global_t_start is None:
                        global_t_start = h['timestamp']
                
                elif Types[evtype] == 'EVTYPE_SCALER' :
                    # TODO
                    pass
                
                elif Types[evtype] == 'EVTYPE_STRON' or \
                     Types[evtype] == 'EVTYPE_STROFF':
                    # EVENTS
                     
                    if code not in allevent:
                        allevent[code] = { }
                    if channel not in allevent[code]:
                        ea = EventArray(name = code , channel_index = channel)
                        # for counting:
                        ea.nbevent = 0
                        ea.maxlabelsize = 0
                        
                        
                        allevent[code][channel] = ea
                        
                    allevent[code][channel].nbevent += 1
                    strobe, = struct.unpack('d' , struct.pack('q' , h['eventoffset']))
                    strobe = str(strobe)
                    if len(strobe)>= allevent[code][channel].maxlabelsize:
                        allevent[code][channel].maxlabelsize = len(strobe)
                    
                    #~ ev = Event()
                    #~ ev.time = h['timestamp'] - global_t_start
                    #~ ev.name = code
                     #~ # it the strobe attribute masked with eventoffset
                    #~ strobe, = struct.unpack('d' , struct.pack('q' , h['eventoffset']))
                    #~ ev.label = str(strobe)
                    #~ seg._events.append( ev )

                elif Types[evtype] == 'EVTYPE_SNIP' :
                    
                    if code not in allspiketr:
                        allspiketr[code] = { }
                    if channel not in allspiketr[code]:
                        allspiketr[code][channel] = { }
                    if h['sortcode'] not in allspiketr[code][channel]:
                        


                        
                        
                        sptr = SpikeTrain([ ], units = 's',
                                                        name = str(h['sortcode']),
                                                        t_start = global_t_start,
                                                        t_stop = global_t_start, # temporary
                                                        left_sweep = (h['size']-10.)/2./h['frequency'] * pq.s,
                                                        sampling_rate = h['frequency'] * pq.Hz,
                                                        
                                                        )
                        #~ sptr.channel = channel
                        #sptr.annotations['channel_index'] = channel
                        sptr.annotate(channel_index = channel)

                        # for counting:
                        sptr.nbspike = 0
                        sptr.pos = 0
                        sptr.waveformsize = h['size']-10
                        
                        #~ sptr.name = str(h['sortcode'])
                        #~ sptr.t_start = global_t_start
                        #~ sptr.sampling_rate = h['frequency']
                        #~ sptr.left_sweep = (h['size']-10.)/2./h['frequency']
                        #~ sptr.right_sweep = (h['size']-10.)/2./h['frequency']
                        #~ sptr.waveformsize = h['size']-10
                        
                        allspiketr[code][channel][h['sortcode']] = sptr
                    
                    allspiketr[code][channel][h['sortcode']].nbspike += 1
                
                elif Types[evtype] == 'EVTYPE_STREAM':
                    if code not in allsig:
                        allsig[code] = { }
                    if channel not in allsig[code]:
                        #~ print 'code', code, 'channel',  channel
                        anaSig = AnalogSignal( 
                                                            [ ] * pq.V,
                                                            name =  code,
                                                            sampling_rate = h['frequency'] * pq.Hz,
                                                            t_start = (h['timestamp'] - global_t_start) * pq.s,
                                                            )
                        
                        anaSig._data_description = {
                                                                    'dtype' : dtype(DataFormats[h['dataformat']]),
                                                                    }
                        #anaSig.annotations['channel_index'] = channel
                        anaSig.annotate(channel_index = channel)
                        anaSig.pos = 0
                        
                        # for counting:
                        anaSig.totalsize = 0
                        #~ anaSig.pos = 0
                        allsig[code][channel] = anaSig
                    allsig[code][channel].totalsize += (h['size']*4-40)/anaSig.dtype.itemsize

            if lazy:
                #TODO : _data_description
                pass
            else:


                # Step 2 : allocate memory
                for code, v in allsig.iteritems():
                    for channel, anaSig in v.iteritems():
                        v[channel] = anaSig.duplicate_with_new_array(np.zeros((anaSig.totalsize) , dtype = anaSig._data_description ['dtype'] )*pq.V )
                        v[channel].pos = 0
                        
                for code, v in allevent.iteritems():
                    for channel, ea in v.iteritems():
                        ea.times = np.empty( (ea.nbevent)  ) * pq.s
                        ea.labels = np.empty( (ea.nbevent), dtype = 'S'+str(ea.maxlabelsize) )
                        ea.pos = 0
                
                for code, v in allspiketr.iteritems():
                    for channel, allsorted in v.iteritems():
                        for sortcode, sptr in allsorted.iteritems():
                            
                            new = SpikeTrain( np.empty( (sptr.nbspike) , dtype = 'f' )*pq.s ,
                                                            name = sptr.name,
                                                            t_start = sptr.t_start,
                                                            t_stop = sptr.t_start + sptr.nbspike*pq.s/float(sptr.sampling_rate.rescale(pq.Hz)),
                                                            left_sweep = sptr.left_sweep,
                                                            sampling_rate = sptr.sampling_rate,
                                                            waveforms = np.empty( (sptr.nbspike, 1, sptr.waveformsize) , dtype = 'f') * pq.mV ,
                                                        )
                            new.annotations.update(sptr.annotations)
                            new.pos = 0
                            new.waveformsize = sptr.waveformsize
                            allsorted[sortcode] = new
                        

                        
                        
                        #~ sptr._spike_times = zeros( (sptr.nbspike), dtype ='f')
                        #~ sptr._waveforms = zeros( (sptr.nbspike, 1 , sptr.waveformsize), dtype = 'f')
            
                # Step 3 : searh sev (individual data files) or tev (common data file)
                # sev is for version > 70
                if os.path.exists(os.path.join(subdir, tankname+'_'+blockname+'.tev')):
                    tev = open(os.path.join(subdir, tankname+'_'+blockname+'.tev'), 'rb')
                else:
                    tev = None
                for code, v in allsig.iteritems():
                    for channel, anaSig in v.iteritems():
                        #~ print anaSig.name, anaSig.channel_index
                        #~ print type(anaSig.name), type(anaSig.channel_index)
                        filename = os.path.join(subdir, tankname+'_'+blockname+'_'+anaSig.name+'_ch'+str(anaSig.annotations['channel_index'])+'.sev')
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
                    channel, code ,  evtype = h['channel'], h['code'], h['evtype']
                    
                    if Types[evtype] == 'EVTYPE_STREAM': 
                        a = allsig[code][channel]
                        dt = a.dtype
                        s = (h['size']*4-40)/dt.itemsize
                        a.fid.seek(h['eventoffset'])
                        #~ print s
                        a[ a.pos:a.pos+s ]  = np.fromstring( a.fid.read( s*dt.itemsize ), dtype = a.dtype)
                        a.pos += s
                    
                    elif Types[evtype] == 'EVTYPE_STRON' or \
                        Types[evtype] == 'EVTYPE_STROFF':
                        ea = allevent[code][channel]
                        ea.times[ea.pos] = (h['timestamp'] - global_t_start) * pq.s
                        strobe, = struct.unpack('d' , struct.pack('q' , h['eventoffset']))
                        ea.labels[ea.pos] = str(strobe)
                        ea.pos += 1
                    
                    elif Types[evtype] == 'EVTYPE_SNIP': 
                        sptr = allspiketr[code][channel][h['sortcode']]
                        sptr[sptr.pos] = h['timestamp'] * pq.s
                        sptr.waveforms[sptr.pos, 0, :] = np.fromstring( sptr.fid.read( sptr.waveformsize*4 ), dtype = 'f4') * pq.V
                        sptr.pos += 1
                
            
            # Step 5 : populating segment
            # FIXME : del attr
            for code, v in allsig.iteritems():
                for channel, anaSig in v.iteritems():
                    #~ del anaSig.totalsize
                    #~ del anaSig.pos
                    #~ del anaSig.fid
                    seg.analogsignals.append( anaSig )

            for code, v in allevent.iteritems():
                for channel, ea in v.iteritems():
                    #~ del ea.nbevent
                    #~ del ea.pos
                    seg.eventarrays.append( ea )


            for code, v in allspiketr.iteritems():
                for channel, allsorted in v.iteritems():
                    for sortcode, sptr in allsorted.iteritems():
                        #~ del sptr.nbspike
                        #~ del sptr.waveformsize
                        #~ del sptr.pos
                        #~ del sptr.fid
                        seg.spiketrains.append( sptr )
        
        create_many_to_one_relationship(bl)
        return bl


TsqDescription = [
    ('size','i'),
    ('evtype','i'),
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
                        0 : np.float32,
                        1 : np.int32,
                        2 : np.int16,
                        3 : np.int8,
                        4 : np.float64,
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




