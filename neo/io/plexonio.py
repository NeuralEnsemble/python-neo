# encoding: utf-8
"""
Class for reading data from Plexion acquisition system (.plx)

Compatible with versions 100 to 106.
Other versions have not been tested.

This IO is developed thanks to the header file downloadable from:
http://www.plexon.com/downloads.html


Depend on: 

Supported : Read

Author: sgarcia

"""

from .baseio import BaseIO
from ..core import *
from .tools import create_many_to_one_relationship, iteritems
import numpy as np
import quantities as pq

import struct
import datetime
import os



class PlexonIO(BaseIO):
    """
    Class for reading plx file.
    
    Usage:
        >>> from neo import io
        >>> r = io.PlexonIO(filename='File_plexon_1.plx')
        >>> seg = r.read_segment(lazy=False, cascade=True)
        >>> print seg.analogsignals
        []
        >>> print seg.spiketrains  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<SpikeTrain(array([  2.75000000e-02,   5.68250000e-02,   8.52500000e-02, ...,
        ...
        >>> print seg.eventarrays
        []
    """
    
    is_readable        = True
    is_writable        = False
    
    supported_objects  = [Segment , AnalogSignal, SpikeTrain, EventArray, EpochArray]
    readable_objects    = [ Segment]
    writeable_objects   = []

    has_header         = False
    is_streameable     = False
    
    # This is for GUI stuf : a definition for parameters when reading.
    read_params        = {
    
                        Segment :  [
                                        ('load_spike_waveform' , { 'value' : False } ) ,
                                        ]
                        }
    write_params       = None
    
    name               = 'Plexon'
    extensions          = [ 'plx' ]
    
    mode = 'file'
    
    
    def __init__(self , filename = None) :
        """
        This class read a plx file.
        
        Arguments:
            filename : the filename
        
        """
        BaseIO.__init__(self)
        self.filename = filename


    def read_segment(self, 
                                        lazy = False,
                                        cascade = True,
                                        load_spike_waveform = False,
                                            ):
        """
        
        """
        
        fid = open(self.filename, 'rb')
        globalHeader = HeaderReader(fid , GlobalHeader ).read_f(offset = 0)
        
        # metadatas
        seg = Segment()
        seg.rec_datetime = datetime.datetime(  globalHeader['Year'] , globalHeader['Month']  , globalHeader['Day'] ,
                    globalHeader['Hour'] , globalHeader['Minute'] , globalHeader['Second'] )
        seg.file_origin = os.path.basename(self.filename)
        seg.annotate(plexon_version = globalHeader['Version'])
        
        if not cascade:
            return seg
        
        
        ## Step 1 : read headers
        
        # dsp channels heade
        dspChannelHeaders = { }
        maxunit=0
        maxchan = 0
        for i in xrange(globalHeader['NumDSPChannels']):
            # channel is 1 based
            channelHeader = HeaderReader(fid , ChannelHeader ).read_f(offset = None)
            channelHeader['Template'] = np.array(channelHeader['Template']).reshape((5,64))
            channelHeader['Boxes'] = np.array(channelHeader['Boxes']).reshape((5,2,4))
            dspChannelHeaders[channelHeader['Channel']]=channelHeader
            maxunit = max(channelHeader['NUnits'],maxunit)
            maxchan = max(channelHeader['Channel'],maxchan)
        
       # event channel header
        eventHeaders = { }
        for i in xrange(globalHeader['NumEventChannels']):
            eventHeader = HeaderReader(fid , EventHeader ).read_f(offset = None)
            eventHeaders[eventHeader['Channel']] = eventHeader

        # slow channel header
        slowChannelHeaders = { }
        for i in xrange(globalHeader['NumSlowChannels']):
            slowChannelHeader = HeaderReader(fid , SlowChannelHeader ).read_f(offset = None)
            slowChannelHeaders[slowChannelHeader['Channel']] = slowChannelHeader
        
        ## Step 2 : prepare allocating
        # for allocating continuous signal
        ncontinuoussamples = np.zeros(len(slowChannelHeaders))
        sampleposition = np.zeros(len(slowChannelHeaders))
        anaSigs = { }
        
        # for allocating spiketimes and waveform
        spiketrains = { }
        nspikecounts = np.zeros((maxchan+1, maxunit+1) ,dtype='i')
        for i,channelHeader in iteritems(dspChannelHeaders):
            spiketrains[i] = { }
        
        # for allocating EventArray
        eventarrays = { }
        neventsperchannel = { }
        #maxstrsizeperchannel = { }
        for chan, h in iteritems(eventHeaders):
            neventsperchannel[chan] = 0
            #maxstrsizeperchannel[chan] = 0
        

        ## Step 3 : a first loop for counting size
        
        
        start = fid.tell()
        while fid.tell() !=-1 :
            # read block header
            dataBlockHeader = HeaderReader(fid , DataBlockHeader ).read_f(offset = None)
            if dataBlockHeader is None : break
            
            chan = dataBlockHeader['Channel']
            unit = dataBlockHeader['Unit']
            n1,n2 = dataBlockHeader['NumberOfWaveforms'] , dataBlockHeader['NumberOfWordsInWaveform']
            
            if dataBlockHeader['Type'] == 1:
                #spike
                if unit not in spiketrains[chan]:
                    sptr = SpikeTrain([ ], units='s', t_stop=0.0)
                    sptr.annotate(unit_name = dspChannelHeaders[chan]['Name'])
                    sptr.annotate(channel_index = i)
                    spiketrains[chan][unit] = sptr
                    
                    spiketrains[chan][unit].sizeOfWaveform = n1,n2
                    
                nspikecounts[chan,unit] +=1
                fid.seek(n1*n2*2,1)
            
            elif dataBlockHeader['Type'] ==4:
                #event
                neventsperchannel[chan] += 1
                if chan not in eventarrays:
                    ea = EventArray()
                    ea.annotate(channel_name= eventHeaders[chan]['Name'])
                    ea.annotate(channel_index = chan)
                    eventarrays[chan] = ea
                
            elif dataBlockHeader['Type'] == 5:
                #continuous signal
                fid.seek(n2*2, 1)
                if n2> 0:
                    ncontinuoussamples[chan] += n2
                    if chan not in anaSigs:
                        anasig =  AnalogSignal(
                                                                        [ ],
                                                                        units = 'V',
                                                                        sampling_rate = float(slowChannelHeaders[chan]['ADFreq'])*pq.Hz,
                                                                        t_start = 0.*pq.s,
                                                                        )
                    anasig.annotate(channel_index = slowChannelHeaders[chan]['Channel'])
                    anasig.annotate(channel_name = slowChannelHeaders[chan]['Name'])
                    anaSigs[chan] =  anasig

        if lazy:
            for chan, anaSig in iteritems(anaSigs):
                anaSigs[chan].lazy_shape = ncontinuoussamples[chan]
                
            for chan, sptrs in iteritems(spiketrains):
                for unit, sptr in iteritems(sptrs):
                    spiketrains[chan][unit].lazy_shape = nspikecounts[chan][unit]
            
            for chan, ea in iteritems(eventarrays):
                ea.lazy_shape = neventsperchannel[chan]
        else:
            ## Step 4: allocating memory if not lazy
            # continuous signal
            for chan, anaSig in iteritems(anaSigs):
                anaSigs[chan] = anaSig.duplicate_with_new_array(np.zeros((ncontinuoussamples[chan]) , dtype = 'f4')*pq.V, )
            
            # allocating mem for SpikeTrain
            for chan, sptrs in iteritems(spiketrains):
                for unit, sptr in iteritems(sptrs):
                        new = SpikeTrain(np.zeros((nspikecounts[chan][unit]), dtype='f')*pq.s, t_stop=1e99) # use an enormous value for t_stop for now, put in correct value later
                        new.annotations.update(sptr.annotations)
                        if load_spike_waveform:
                            n1, n2 = spiketrains[chan][unit].sizeOfWaveform
                            new.waveforms = np.zeros( (nspikecounts[chan][unit], n1, n2 )*pq.V , dtype = 'f' ) * pq.V
                        spiketrains[chan][unit] = new
            nspikecounts[:] = 0
            
            # event
            eventpositions = { }
            for chan, ea in iteritems(eventarrays):
                ea.times = np.zeros( neventsperchannel[chan] )*pq.s
                #ea.labels = zeros( neventsperchannel[chan] , dtype = 'S'+str(neventsperchannel[chan]) )
                eventpositions[chan]=0
            
        if not lazy:
            
            ## Step 5 : a second loop for reading if not lazy
            fid.seek(start)
            while fid.tell() !=-1 :
                dataBlockHeader = HeaderReader(fid , DataBlockHeader ).read_f(offset = None)
                if dataBlockHeader is None : break
                chan = dataBlockHeader['Channel']
                n1,n2 = dataBlockHeader['NumberOfWaveforms'] , dataBlockHeader['NumberOfWordsInWaveform']
                time = dataBlockHeader['UpperByteOf5ByteTimestamp']*2.**32 + dataBlockHeader['TimeStamp']
                time/= globalHeader['ADFrequency'] 
                
                if n2 <0: break
                if dataBlockHeader['Type'] == 1:
                    #spike
                    unit = dataBlockHeader['Unit']
                    sptr = spiketrains[chan][unit]
                    
                    pos = nspikecounts[chan,unit]
                    sptr[pos] = time * pq.s
                    
                    if load_spike_waveform and n1*n2 != 0 :
                        waveform = fromstring( fid.read(n1*n2*2) , dtype = 'i2').reshape(n1,n2).astype('f')
                        #range
                        if globalHeader['Version'] <103:
                            waveform = waveform*3000./(2048*dspChannelHeaders[chan]['Gain']*1000.)
                        elif globalHeader['Version'] >=103 and globalHeader['Version'] <105:
                            waveform = waveform*globalHeader['SpikeMaxMagnitudeMV']/(.5*2.**(globalHeader['BitsPerSpikeSample'])*1000.)
                        elif globalHeader['Version'] >105:
                            waveform = waveform*globalHeader['SpikeMaxMagnitudeMV']/(.5*2.**(globalHeader['BitsPerSpikeSample'])*globalHeader['SpikePreAmpGain'])
                        
                        sptr._waveforms[pos,:,:] = waveform
                    else:
                        fid.seek(n1*n2*2,1)
                    
                    nspikecounts[chan,unit] +=1
                    
                        
                    
                    
                elif dataBlockHeader['Type'] == 4:
                    # event
                    pos = eventpositions[chan]
                    eventarrays[chan].times[pos] = time * pq.s
                    eventpositions[chan]+= 1
                
                elif dataBlockHeader['Type'] == 5:
                    #signal
                    data = np.fromstring( fid.read(n2*2) , dtype = 'i2').astype('f4')
                    #range
                    if globalHeader['Version'] ==100 or globalHeader['Version'] ==101 :
                        data = data*5000./(2048*slowChannelHeaders[chan]['Gain']*1000.)
                    elif globalHeader['Version'] ==102 :
                        data = data*5000./(2048*slowChannelHeaders[chan]['Gain']*slowChannelHeaders[chan]['PreampGain'])
                    elif globalHeader['Version'] >= 103:
                        data = data*globalHeader['SlowMaxMagnitudeMV']/(.5*(2**globalHeader['BitsPerSpikeSample'])*\
                                                            slowChannelHeaders[chan]['Gain']*slowChannelHeaders[chan]['PreampGain'])
                    anaSigs[chan][sampleposition[chan] : sampleposition[chan]+data.size] = data * pq.V
                    sampleposition[chan] += data.size
                    if sampleposition[chan] ==0:
                        anaSigs[chan].t_start = time* pq.s
                
        
            
        
        #TODO if lazy
        
        
        # add AnalogSignal to sgement
        for k,anaSig in iteritems(anaSigs) :
            if anaSig is not None:
                seg.analogsignals.append(anaSig)
                
        # add SpikeTrain to sgement
        for chan, sptrs in iteritems(spiketrains):
            for unit, sptr in iteritems(sptrs):
                if len(sptr) > 0:
                    sptr.t_stop = sptr.max() # can probably get a better value for this, from the associated AnalogSignal
                seg.spiketrains.append(sptr)
        
        # add eventarray to segment
        for chan,ea in  iteritems(eventarrays):
            seg.eventarrays.append(ea)
        
        create_many_to_one_relationship(seg)
        return seg



GlobalHeader = [
    ('MagicNumber' , 'I'),
    ('Version','i'),
    ('Comment','128s'),
    ('ADFrequency','i'),
    ('NumDSPChannels','i'),
    ('NumEventChannels','i'),
    ('NumSlowChannels','i'),
    ('NumPointsWave','i'),
    ('NumPointsPreThr','i'),
    ('Year','i'),
    ('Month','i'),
    ('Day','i'),
    ('Hour','i'),
    ('Minute','i'),
    ('Second','i'),
    ('FastRead','i'),
    ('WaveformFreq','i'),
    ('LastTimestamp','d'),
    
    #version >103
    ('Trodalness' , 'b'),
    ('DataTrodalness' , 'b'),
    ('BitsPerSpikeSample' , 'b'),
    ('BitsPerSlowSample' , 'b'),
    ('SpikeMaxMagnitudeMV' , 'H'),
    ('SlowMaxMagnitudeMV' , 'H'),
    
    #version 105
    ('SpikePreAmpGain' , 'H'),
    
    #version 106
    ('AcquiringSoftware','18s'),
    ('ProcessingSoftware','18s'),

    ('Padding','10s'),
    
    # all version
    ('TSCounts','650i'),
    ('WFCounts','650i'),
    ('EVCounts','512i'),
    
    ]


ChannelHeader = [
    ('Name' , '32s'),
    ('SIGName','32s'),
    ('Channel','i'),
    ('WFRate','i'),
    ('SIG','i'),
    ('Ref','i'),
    ('Gain','i'),
    ('Filter','i'),
    ('Threshold','i'),
    ('Method','i'),
    ('NUnits','i'),
    ('Template','320h'),
    ('Fit','5i'),
    ('SortWidth','i'),
    ('Boxes','40h'),
    ('SortBeg','i'),
    #version 105
    ('Comment','128s'),
    #version 106
    ('SrcId','b'),
    ('reserved','b'),
    ('ChanId','H'),
    
    ('Padding','10i'),
    ]

EventHeader = [
    ('Name' , '32s'),
    ('Channel','i'),
    #version 105
    ('Comment' , '128s'),
    #version 106
    ('SrcId','b'),
    ('reserved','b'),
    ('ChanId','H'),
    
    ('Padding','32i'),
    ]


SlowChannelHeader = [
    ('Name' , '32s'),
    ('Channel','i'),
    ('ADFreq','i'),
    ('Gain','i'),
    ('Enabled','i'),
    ('PreampGain','i'),
    #version 104
    ('SpikeChannel','i'),
    #version 105
    ('Comment','128s'),
    #version 106
    ('SrcId','b'),
    ('reserved','b'),
    ('ChanId','H'),
    
    ('Padding','27i'),
    ]

DataBlockHeader = [
    ('Type','h'),
    ('UpperByteOf5ByteTimestamp','h'),
    ('TimeStamp','i'),
    ('Channel','h'),
    ('Unit','h'),
    ('NumberOfWaveforms','h'),
    ('NumberOfWordsInWaveform','h'),
    ]# 16 bytes


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



