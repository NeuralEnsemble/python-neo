# encoding: utf-8
"""
Classe for reading data in CED spike2 files (.smr).

This code is based on:
 - sonpy, written by Antonio Gonzalez <Antonio.Gonzalez@cantab.net>
    Disponible here ::
    http://www.neuro.ki.se/broberger/

and sonpy come from :
 - SON Library 2.0 for MATLAB, written by Malcolm Lidierth at
    King's College London. See http://www.kcl.ac.uk/depsta/biomedical/cfnr/lidierth.html

This IO support old (<v6) and new files (>v7) of spike2


Depend on: 

Supported : Read

Author: sgarcia

"""

from .baseio import BaseIO
from ..core import *
from .tools import create_many_to_one_relationship

import numpy as np
from numpy import dtype, zeros, fromstring, empty
import quantities as pq

import os, sys

PY3K = (sys.version_info[0] == 3)


class Spike2IO(BaseIO):
    """
    Class for reading data from CED spike2.
    
    Usage:
        >>> from neo import io
        >>> r = io.Spike2IO( filename = 'File_spike2_1.smr')
        >>> seg = r.read_segment(lazy = False, cascade = True,)
        >>> print seg.analogsignals
        >>> print seg.spiketrains
        >>> print seg.eventarrays
    
    """
    
    is_readable        = True
    is_writable        = False
    
    supported_objects  = [ Segment , AnalogSignal , EventArray, SpikeTrain]
    readable_objects   = [Segment]
    writeable_objects  = [ ]      


    has_header         = False
    is_streameable     = False
    read_params        = {   Segment : [ ], }
    write_params       = None

    name               = 'Spike 2 CED'
    extensions          = [ 'smr' ]
    
    mode = 'file'
    
    def __init__(self , filename = None) :
        """
        This class read a smr file.
        
        Arguments:
            filename : the filename 
        """
        BaseIO.__init__(self)
        self.filename = filename

    def read_segment(self ,
                                            lazy = False,
                                            cascade = True,

                                                ):
        """
        Arguments:
        """
        
        
        header = self.read_header(filename = self.filename)
        
        #~ print header
        fid = open(self.filename, 'rb')
        
        seg  = Segment(
                                    file_origin = os.path.basename(self.filename),
                                    ced_version = str(header.system_id),
                                    )
        
        if not cascade:
            return seg
        
        def addannotations(ob, channelHeader):
            ob.annotate(title = channelHeader.title)
            ob.annotate(physical_channel_index = channelHeader.phy_chan)
            ob.annotate(comment = channelHeader.comment)
        
        for i in range(header.channels) :
            channelHeader = header.channelHeaders[i]
            
            #~ print 'channel' , i , 'kind' ,  channelHeader.kind
            
            if channelHeader.kind !=0:
                #~ print '####'
                #~ print 'channel' , i, 'kind' , channelHeader.kind , channelHeader.type , channelHeader.phy_chan
                #~ print channelHeader
                pass
            
            if channelHeader.kind in [1, 9]:
                #~ print 'analogChanel'
                anaSigs = self.readOneChannelContinuous( fid, i, header ,lazy = lazy)
                #~ print 'nb sigs', len(anaSigs) , ' sizes : ',
                for anaSig in anaSigs :
                    addannotations(anaSig, channelHeader)
                    seg.analogsignals.append( anaSig )
                    #~ print sig.signal.size,
                #~ print ''
                    
            elif channelHeader.kind in  [2, 3, 4, 5, 8] :
                ea = self.readOneChannelEventOrSpike( fid, i, header , lazy = lazy)
                addannotations(ea, channelHeader)
                seg.eventarrays.append(ea)
                
            elif channelHeader.kind in  [6,7] :
                sptr = self.readOneChannelEventOrSpike( fid, i, header, lazy = lazy )
                if sptr is not None:
                    addannotations(sptr, channelHeader)
                    seg.spiketrains.append(sptr)
            
        fid.close()
        
        create_many_to_one_relationship(seg)
        return seg
        
        
    def read_header(self , filename = ''):
        
        fid = open(filename, 'rb')
        header = HeaderReader(fid,   dtype(headerDescription))
        #~ print 'chan_size' , header.chan_size
        
        
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
                      ('unit' , 'S6'),]
                channelHeader += HeaderReader(fid, dtype(dt))
                if header.system_id < 6:
                    channelHeader += HeaderReader(fid, dtype([ ('divide' , 'i4')]) )#i8
                else : 
                    channelHeader +=HeaderReader(fid, dtype([ ('interleave' , 'i4')]) )#i8
            if channelHeader.kind in [7, 9]:
                dt = [('min' , 'f4'),
                      ('max' , 'f4'),
                      ('unit' , 'S6'),]
                channelHeader += HeaderReader(fid, dtype(dt))
                if header.system_id < 6:
                    channelHeader += HeaderReader(fid, dtype([ ('divide' , 'i4')]))#i8
                else :
                    channelHeader += HeaderReader(fid, dtype([ ('interleave' , 'i4')]) )#i8
            if channelHeader.kind in [4]:
                dt = [('init_low' , 'u1'),
                      ('next_low' , 'u1'),]
                channelHeader += HeaderReader(fid, dtype(dt))
            
            channelHeader.type = dict_kind[channelHeader.kind]
            channelHeaders.append(channelHeader)
            
        header.channelHeaders = channelHeaders
        
        fid.close()
        return header

    
    def readOneChannelContinuous(self , fid, channel_num, header ,lazy = True):
        # read AnalogSignal
        channelHeader = header.channelHeaders[channel_num]
        
        
        # data type
        if channelHeader.kind == 1:
            dt = np.dtype('i2')
        elif channelHeader.kind == 9:
            dt = np.dtype('f4')
        
        # sample rate
        if header.system_id in [1,2,3,4,5]: # Before version 5
            sample_interval = (channelHeader.divide*header.us_per_time*header.time_per_adc)*1e-6
        else :
            sample_interval = (channelHeader.l_chan_dvd*header.us_per_time*header.dtime_base)
        sampling_rate = (1./sample_interval)*pq.Hz
        
        # read blocks header to preallocate memory by jumping block to block
        fid.seek(channelHeader.firstblock)
        blocksize = [ 0 ]
        starttimes = [ ]
        for b in range(channelHeader.blocks) :
            blockHeader = HeaderReader(fid, dtype(blockHeaderDesciption))
            if len(blocksize) > len(starttimes):
                starttimes.append(blockHeader.start_time)
            blocksize[-1] += blockHeader.items
            
            if blockHeader.succ_block > 0 :
                # this is ugly but CED do not garanty continuity in AnalogSignal
                fid.seek(blockHeader.succ_block)
                nextBlockHeader = HeaderReader(fid, dtype(blockHeaderDesciption))
                sample_interval = (blockHeader.end_time-blockHeader.start_time)/(blockHeader.items-1)
                interval_with_next = nextBlockHeader.start_time - blockHeader.end_time
                if interval_with_next > sample_interval:
                    blocksize.append(0)
                fid.seek(blockHeader.succ_block)
        
        anaSigs = [ ]
        if channelHeader.unit in unit_convert:
            unit = pq.Quantity(1, unit_convert[channelHeader.unit] )
        else:
            #print channelHeader.unit
            try:
                unit = pq.Quantity(1, channelHeader.unit )
            except:
                unit = pq.Quantity(1, '')
            
        for b,bs in enumerate(blocksize ):
            if lazy:
                signal = [ ]*unit
            else:
                signal = empty( bs , dtype = 'f4') * unit
            anaSig = AnalogSignal(signal ,
                                                            sampling_rate = sampling_rate,
                                                            t_start = starttimes[b]*header.us_per_time * header.dtime_base * pq.s,
                                                            )
            anaSig.annotate(channel_index = channel_num)
            anaSigs.append( anaSig )
        
        if  lazy:
            for s, anaSig in enumerate(anaSigs):
                anaSig.lazy_shape = blocksize[s]
            
        else:
            # read data  by jumping block to block
            fid.seek(channelHeader.firstblock)
            pos = 0
            numblock = 0
            for b in range(channelHeader.blocks) :
                blockHeader = HeaderReader(fid, dtype(blockHeaderDesciption))
                # read data
                sig = fromstring( fid.read(blockHeader.items*dt.itemsize) , dtype = dt)
                anaSigs[numblock][pos:pos+sig.size] = sig.astype('f4')*unit
                pos += sig.size
                if pos >= blocksize[numblock] :
                    numblock += 1
                    pos = 0
                # jump to next block
                if blockHeader.succ_block > 0 :
                    fid.seek(blockHeader.succ_block)
            
        # convert for int16
        if dt.kind == 'i' :
            for anaSig in anaSigs :
                anaSig *= channelHeader.scale/ 6553.6
                anaSig += channelHeader.offset*unit
        
        return anaSigs
    
    
    def readOneChannelEventOrSpike(self , fid, channel_num, header ,lazy = True):
        # return SPikeTrain or EventArray
        channelHeader = header.channelHeaders[channel_num]
        if channelHeader.firstblock <0: return
        if channelHeader.kind not in [2, 3, 4 , 5 , 6 ,7, 8]: return
        
        ## Step 1 : type of blocks
        if channelHeader.kind in [2, 3, 4]:
            # Event data
            format = [('tick' , 'i4') ]
        elif channelHeader.kind in [5]:
            # Marker data
            format = [('tick' , 'i4') , ('marker' , 'i4') ]
        elif channelHeader.kind in [6]:
            # AdcMark data
            format = [('tick' , 'i4') , ('marker' , 'i4')  , ('adc' , 'S%d' %channelHeader.n_extra   )]
        elif channelHeader.kind in [7]:
            #  RealMark data
            format = [('tick' , 'i4') , ('marker' , 'i4')  , ('real' , 'S%d' %channelHeader.n_extra   )]
        elif channelHeader.kind in [8]:
            # TextMark data
            format = [('tick' , 'i4') , ('marker' , 'i4')  ,  ('label' , 'S%d'%channelHeader.n_extra)]
        dt = dtype(format)
        
            
        ## Step 2 : first read for allocating mem
        fid.seek(channelHeader.firstblock)
        totalitems = 0
        for b in range(channelHeader.blocks) :
            blockHeader = HeaderReader(fid, dtype(blockHeaderDesciption))
            totalitems += blockHeader.items
            if blockHeader.succ_block > 0 :
                fid.seek(blockHeader.succ_block)
        #~ print 'totalitems' , totalitems
        
        if lazy :
            if channelHeader.kind in [2, 3, 4 , 5 , 8]:
                ea = EventArray(  )
                ea.annotate(channel_index = channel_num)
                ea.lazy_shape = totalitems
                return ea
                
            elif channelHeader.kind in [6 ,7]:
                sptr = SpikeTrain([ ]*pq.s, t_stop=1e99)  # correct value for t_stop to be put in later
                sptr.annotate(channel_index = channel_num)
                sptr.lazy_shape = totalitems
                return sptr
        
        else:
            alltrigs = zeros( totalitems , dtype = dt)
            ## Step 3 : read
            fid.seek(channelHeader.firstblock)
            pos = 0
            for b in range(channelHeader.blocks) :
                blockHeader = HeaderReader(fid, dtype(blockHeaderDesciption))
                # read all events in block
                trigs = fromstring( fid.read( blockHeader.items*dt.itemsize)  , dtype = dt)
                
                alltrigs[pos:pos+trigs.size] = trigs
                pos += trigs.size
                if blockHeader.succ_block > 0 :
                    fid.seek(blockHeader.succ_block)
            
            ## Step 3 convert in neo standard class : eventarrays or spiketrains
            alltimes = alltrigs['tick'].astype('f')*header.us_per_time * header.dtime_base*pq.s
            
            if channelHeader.kind in [2, 3, 4 , 5 , 8]:
                #events
                ea = EventArray(  )
                ea.annotate(channel_index = channel_num)
                ea.times = alltimes
                if channelHeader.kind >= 5:
                    # Spike2 marker is closer to label sens of neo
                    ea.labels = alltrigs['marker'].astype('S')
                if channelHeader.kind == 8:
                    ea.annotate(extra_labels = alltrigs['label'])
                return ea
                
            elif channelHeader.kind in [6 ,7]:
                # spiketrains
                
                # waveforms
                if channelHeader.kind == 6 :
                    waveforms = fromstring(alltrigs['adc'].tostring() , dtype = 'i2')
                    waveforms = waveforms.astype('f4') *channelHeader.scale/ 6553.6 + channelHeader.offset
                elif channelHeader.kind == 7 :
                    waveforms = fromstring(alltrigs['real'].tostring() , dtype = 'f4')
                
                
                if header.system_id>=6 and channelHeader.interleave>1:
                    waveforms = waveforms.reshape((alltimes.size,-1,channelHeader.interleave))
                    waveforms = waveforms.swapaxes(1,2)
                else:
                    waveforms = waveforms.reshape(( alltimes.size,1, -1))
                
                
                if header.system_id in [1,2,3,4,5]:
                    sample_interval = (channelHeader.divide*header.us_per_time*header.time_per_adc)*1e-6
                else :
                    sample_interval = (channelHeader.l_chan_dvd*header.us_per_time*header.dtime_base)
                
                if channelHeader.unit in unit_convert:
                    unit = pq.Quantity(1, unit_convert[channelHeader.unit] )
                else:
                    #print channelHeader.unit
                    try:
                        unit = pq.Quantity(1, channelHeader.unit )
                    except:
                        unit = pq.Quantity(1, '')
                
                if len(alltimes) > 0:
                    t_stop = alltimes.max() # can get better value from associated AnalogSignal(s) ?
                else:
                    t_stop = 0.0
                sptr = SpikeTrain(alltimes,
                                            waveforms = waveforms*unit,
                                            sampling_rate = (1./sample_interval)*pq.Hz,
                                            t_stop = t_stop
                                            )
                sptr.annotate(channel_index = channel_num)
                
                return sptr
            
            




class HeaderReader(object):
    def __init__(self , fid , dtype):
        if fid is not None :
            array = np.fromstring( fid.read(dtype.itemsize) , dtype)[0]
        else :
            array = zeros( (1) , dtype = dtype)[0]
        object.__setattr__(self, 'dtype' , dtype)
        object.__setattr__(self, 'array' , array)
        
    def __setattr__(self, name , val):
        if name in self.dtype.names :
            self.array[name] = val
        else :
            object.__setattr__(self, name , val)

    def __getattr__(self , name):
        #~ print name
        if name in self.dtype.names :
            if self.dtype[name].kind == 'S':
                if PY3K:
                    l = np.fromstring(self.array[name].decode('iso-8859-1')[0], 'u1')    
                else:
                    l = np.fromstring(self.array[name][0], 'u1')
                return self.array[name][1:l+1]
            else:
                return self.array[name]
        else :
            object.__getattr__(self, name )
    def names(self):
        return self.array.dtype.names
    
    def __repr__(self):
        s = 'HEADER'
        for name in self.dtype.names :
            #~ if self.dtype[name].kind != 'S' :    
                s += name + self.__getattr__(name)
        return s
    
                
    
    def __add__(self, header2):
#        print 'add' , self.dtype, header2.dtype
        newdtype = [ ]
        for name in self.dtype.names :
            newdtype.append( (name , self.dtype[name].str) )
        for name in header2.dtype.names :
            newdtype.append( (name , header2.dtype[name].str) )
        newdtype = dtype(newdtype)
        newHeader = HeaderReader(None , newdtype )
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
    ('max_chan_time','i4'),#i8
    ('l_chan_dvd','i4'),#i8
    ('phy_chan','i2'),
    ('title','S10'),
    ('ideal_rate','f4'),
    ('kind','u1'),
    ('unused1','i1'),
    
    ]

dict_kind = {
    0 : 'empty',
    1: 'Adc',
    2: 'EventFall',
    3: 'EventRise',
    4: 'EventBoth',
    5: 'Marker',
    6: 'AdcMark',
    7: 'RealMark',
    8: 'TextMark',
    9: 'RealWave',
    }


blockHeaderDesciption =[
    ('pred_block','i4'),#i8
    ('succ_block','i4'),#i8
    ('start_time','i4'),#i8
    ('end_time','i4'),#i8
    ('channel_num','i2'),
    ('items','i2'),
    ]


unit_convert = {
                        'Volts' : 'V' ,
                        }