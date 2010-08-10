# -*- coding: utf-8 -*-
"""

Classe for reading data from Plexion acquisition system (.plx)

Compatible from version 100 to 106.
Other versions are not controled.

This IO is develloped thanks to the header file downloadable here:
http://www.plexon.com/downloads.html


Supported : Read

@author :  luc estebanez ,sgarcia


"""






from baseio import BaseIO
#from neo.core import *
from ..core import *

from numpy import *
import struct
import datetime

class PlexonIO(BaseIO):
    """
    Class for reading/writing data in a fake file.
    
    **Usage**

    **Example**
    
    """
    
    is_readable        = True
    is_writable        = False
    
    supported_objects  = [Segment , AnalogSignal, SpikeTrain, Event, Epoch]
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
    
    filemode = True
    

    
    def __init__(self , filename = None) :
        """
        This class read a plx file.
        
        **Arguments**
        
            filename : the filename to read you can pu what ever it do not read anythings
        
        """
        BaseIO.__init__(self)
        self.filename = filename


    def read(self , **kargs):
        """
        Read a fake file.
        Return a neo.Segment
        See read_block for detail.
        """
        return self.read_segment( **kargs)



    def read_segment(self, load_spike_waveform = False):
        """
        """
        seg = Segment()
        fid = open(self.filename, 'rb')
        globalHeader = HeaderReader(fid , GlobalHeader ).read_f(offset = 0)
        #~ globalHeader['TSCounts'] = array(globalHeader['TSCounts']).reshape((130,5))
        #~ globalHeader['WFCounts'] = array(globalHeader['WFCounts']).reshape((130,5))
        
        
        seg.filedatetime = datetime.datetime(  globalHeader['Year'] , globalHeader['Month']  , globalHeader['Day'] ,
                    globalHeader['Hour'] , globalHeader['Minute'] , globalHeader['Second'] )
        
        
        #~ print 'version' , globalHeader['Version']
        seg.plexonVersion = globalHeader['Version']
        
        ## Step 1 : read headers
        
        # dsp channels heade
        dspChannelHeaders = { }
        maxunit=0
        maxchan = 0
        for i in xrange(globalHeader['NumDSPChannels']):
            # channel is 1 based
            channelHeader = HeaderReader(fid , ChannelHeader ).read_f(offset = None)
            channelHeader['Template'] = array(channelHeader['Template']).reshape((5,64))
            channelHeader['Boxes'] = array(channelHeader['Boxes']).reshape((5,2,4))
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
        ncontinuoussamples = zeros(len(slowChannelHeaders))
        sampleposition = zeros(len(slowChannelHeaders))
        anaSigs = { }
        
        # for allocating spiketimes and waveform
        spiketrains = { }
        nspikecounts = zeros((maxchan+1, maxunit+1) ,dtype='i')
        for i,channelHeader in dspChannelHeaders.iteritems():
            
            spiketrains[i] = { }


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
                    spiketrains[chan][unit] = SpikeTrain(
                                                                            channel = i,
                                                                            name = dspChannelHeaders[chan]['Name'],
                                                                            sampling_rate = globalHeader['ADFrequency'],
                                                                            )
                    spiketrains[chan][unit].sizeOfWaveform = n1,n2
                nspikecounts[chan,unit] +=1
                fid.seek(n1*n2*2,1)
            
            elif dataBlockHeader['Type'] ==4:
                #event
                pass
            
            elif dataBlockHeader['Type'] == 5:
                #continuous signal
                fid.seek(n2*2, 1)
                if n2> 0:
                    ncontinuoussamples[chan] += n2
                    if chan not in anaSigs:
                        anaSigs[chan] =  AnalogSignal(
                                                                        sampling_rate = float(slowChannelHeaders[chan]['ADFreq']),
                                                                        t_start = 0.,
                                                                        name = slowChannelHeaders[chan]['Name'],
                                                                        channel = slowChannelHeaders[chan]['Channel']
                                                                        )
            else :
                pass
        
        ## Step 4: allocating memory 
        
        # continuous signal
        for chan, anaSig in anaSigs.iteritems():
            anaSig.signal = zeros((ncontinuoussamples[chan]) , dtype = 'f4')
        
        # allocating mem for SpikeTrain
        for chan, sptrs in spiketrains.iteritems():
            for unit, sptr in sptrs.iteritems():
                    spiketrains[chan][unit]._spike_times = zeros( (nspikecounts[chan][unit]) , dtype = 'f' )
                    if load_spike_waveform:
                        n1, n2 = spiketrains[chan][unit].sizeOfWaveform
                        spiketrains[chan][unit]._waveforms = zeros( (nspikecounts[chan][unit], n1, n2 ) , dtype = 'f' )
        nspikecounts[:] = 0
        
        ## Step 5 : a second loop for reading
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
                sptr._spike_times[pos] = time
                
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
                ev = Event(time = time)
                ev.name = eventHeaders[chan]['Name']
                seg._events.append(ev)
            
            elif dataBlockHeader['Type'] == 5:
                #signal
                data = fromstring( fid.read(n2*2) , dtype = 'i2').astype('f4')
                #range
                if globalHeader['Version'] ==100 or globalHeader['Version'] ==101 :
                    data = data*5000./(2048*slowChannelHeaders[chan]['Gain']*1000.)
                elif globalHeader['Version'] ==102 :
                    data = data*5000./(2048*slowChannelHeaders[chan]['Gain']*slowChannelHeaders[chan]['PreampGain'])
                elif globalHeader['Version'] >= 103:
                    data = data*globalHeader['SlowMaxMagnitudeMV']/(.5*(2**globalHeader['BitsPerSpikeSample'])*\
                                                        slowChannelHeaders[chan]['Gain']*slowChannelHeaders[chan]['PreampGain'])
                anaSigs[chan].signal[sampleposition[chan] : sampleposition[chan]+data.size] = data
                sampleposition[chan] += data.size
                if sampleposition[chan] ==0:
                    anaSigs[chan].t_start = time
            
            else :
                pass
                
        # add AnalogSignal to sgement
        for k,anaSig in anaSigs.iteritems() :
            if anaSig is not None:
                seg._analogsignals.append(anaSig)
                
        # add SpikeTrain to sgement
        for chan, sptrs in spiketrains.iteritems():
            for unit, sptr in sptrs.iteritems():
                    seg._spiketrains.append(sptr)

        
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



