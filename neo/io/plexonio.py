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
        globalHeader['TSCounts'] = array(globalHeader['TSCounts']).reshape((130,5))
        globalHeader['WFCounts'] = array(globalHeader['WFCounts']).reshape((130,5))
        TSCounts = globalHeader['TSCounts']
        WFCounts = globalHeader['WFCounts']
        for k,v in globalHeader.iteritems():
            #~ if type(v) != numpy.ndarray : print k,v
            print k,v
        
        seg.filedatetime = datetime.datetime(  globalHeader['Year'] , globalHeader['Month']  , globalHeader['Day'] ,
                    globalHeader['Hour'] , globalHeader['Minute'] , globalHeader['Second'] )
        
        print 'version' , globalHeader['Version']
        print 
        
        channelHeaders = { }
        # dsp channels header
        for i in xrange(globalHeader['NumDSPChannels']):
            
            channelHeader = HeaderReader(fid , ChannelHeader ).read_f(offset = None)
            channelHeader['Template'] = array(channelHeader['Template']).reshape((5,64))
            channelHeader['Boxes'] = array(channelHeader['Boxes']).reshape((5,2,4))
            channelHeaders[channelHeader['Channel']]=channelHeader
            #~ print 'i',i
            #~ for k,v in channelHeader.iteritems(): print k,v
        #~ print ''
        #~ print 'channelHeaders' , len(channelHeaders)
        #~ for k,v in channelHeaders.iteritems(): print k,v
        
        
        
        
        # event channel header
        eventHeaders = { }
        for i in xrange(globalHeader['NumEventChannels']):
            #~ print ''
            #~ print 'channel events',i
            #~ print 'tell' , fid.tell()
            eventHeader = HeaderReader(fid , EventHeader ).read_f(offset = None)
            #~ for k,v in eventHeader.iteritems(): print k,v
            eventHeaders[eventHeader['Channel']] = eventHeader
        #~ print ''
        #~ print 'eventHeaders' , len(eventHeaders)
        #~ for k,v in eventHeaders.iteritems(): print k,v

        
        slowChannelHeaders = { }
        # slow channel header
        for i in xrange(globalHeader['NumSlowChannels']):
            #~ print 'i',i
            slowChannelHeader = HeaderReader(fid , SlowChannelHeader ).read_f(offset = None)
            slowChannelHeaders[slowChannelHeader['Channel']] = slowChannelHeader
            #~ for k,v in slowChannelHeader.iteritems(): print k,v
        #~ print ''
        #~ print 'slowChannelHeaders' , len(slowChannelHeaders)
        #~ for k,v in slowChannelHeaders.iteritems(): print k,v
        
        # for post allocating continuous signal
        ncontinuoussamples = zeros(len(slowChannelHeaders))
        sampleposition = zeros(len(slowChannelHeaders))
        
        # allocate spiketimes and waveform
        spiketrains = [ ]
        for i in xrange(TSCounts.shape[0]):
            spiketrains.append([])
            for j in xrange(5):
                if load_spike_waveform and WFCounts[i,j] != 0:
                    sptr = SpikeTrain(spikes = [ ])
                    sptr.sampling_rate = globalHeader['ADFrequency']
                    spiketrains[-1].append(sptr  )
                elif TSCounts[i,j] !=0:
                    sptr = SpikeTrain(spike_times = zeros((TSCounts[i,j]) , dtype='f4'))
                    sptr.channel = i
                    sptr.sampling_rate = globalHeader['ADFrequency']
                    spiketrains[-1].append( sptr )
                    nspikecounts = zeros(TSCounts.shape ,dtype='i')
                else :
                    spiketrains[-1].append(None)
            else :
                spiketrains[-1].append(None)
        
        # data block header
        # loop on the data blocks  : first parts for spike and events
        # a second loop next for continuous data after allocated memory
        start = fid.tell()
        while fid.tell() !=-1 :
            dataBlockHeader = HeaderReader(fid , DataBlockHeader ).read_f(offset = None)
            #~ print ''
            #~ for k,v in dataBlockHeader.iteritems(): print k,v
            
            if dataBlockHeader is None : break
            chan = dataBlockHeader['Channel']
            unit = dataBlockHeader['Unit']
            time = dataBlockHeader['UpperByteOf5ByteTimestamp']*2.**32 + dataBlockHeader['TimeStamp']
            time/= globalHeader['ADFrequency'] 
            n1,n2 = dataBlockHeader['NumberOfWaveforms'] , dataBlockHeader['NumberOfWordsInWaveform']
            
            if dataBlockHeader['Type'] == 1:
                #spike
                #~ print ''
                #~ for k,v in dataBlockHeader.iteritems(): print k,v
                
                if load_spike_waveform and n1*n2 != 0 :
                    data = fromstring( fid.read(n1*n2*2) , dtype = 'i2').reshape(n1,n2).astype('f')
                    #range
                    if globalHeader['Version'] <103:
                        data = data*3000./(2048*channelHeaders[chan]['Gain']*1000.)
                    elif globalHeader['Version'] >=103 and globalHeader['Version'] <105:
                        data = data*globalHeader['SpikeMaxMagnitudeMV']/(.5*2.**(globalHeader['BitsPerSpikeSample'])*1000.)
                    elif globalHeader['Version'] >105:
                        data = data*globalHeader['SpikeMaxMagnitudeMV']/(.5*2.**(globalHeader['BitsPerSpikeSample'])*globalHeader['SpikePreAmpGain'])
                    sp = Spike(time = time,
                                sampling_rate = channelHeaders[chan]['WFRate'],
                                waveform = data)
                    spiketrains[chan][unit]._spikes.append(sp)
                else :
                    pos = nspikecounts[chan,unit]
                    #~ print 'pos', pos
                    spiketrains[chan][unit]._spike_times[pos] = time
                    nspikecounts[chan,unit] +=1
                    fid.seek(n1*n2*2,1)
                    
            elif dataBlockHeader['Type'] ==4:
                #event
                #~ print ''
                #~ for k,v in dataBlockHeader.iteritems(): print k,v
                ev = Event(time = time)
                ev.name = eventHeaders[chan]['Name']
                seg._events.append(ev)
            
            elif dataBlockHeader['Type'] == 5:
                #continuous : not read here just counting n samples
                
                #~ print ''
                #~ for k,v in dataBlockHeader.iteritems(): print k,v
                #~ data = fromstring( fid.read(dataBlockHeader['NumberOfWordsInWaveform']*2) , dtype = 'i2').astype('f')
                
                #fid.seek(fid.tell()+dataBlockHeader['NumberOfWordsInWaveform']*2)
                fid.seek(n2*2, 1)
                if n1 !=1 :
                    print 'probable bug because NumberOfWaveforms = ',n1, ' should be 1 at fid.tell()', fid.tell()
                if n2> 0:
                    ncontinuoussamples[chan] += n2
                else :
                    print 'probable bug because NumberOfWordsInWaveform = ',n2, 'at fid.tell()', fid.tell()
            else :
                print 'probable bug because unkonwn type = ',dataBlockHeader['Type'], 'at fid.tell()', fid.tell()
                pass
                #~ print 'unkonwn type',dataBlockHeader['Type']
                #~ print '# tell #', fid.tell()
                break
             #~ print ''
                #~ for k,v in dataBlockHeader.iteritems(): print k,v
                
        # continuous : allcating mem
        anaSigs = { }
        for i in range(ncontinuoussamples.size):
            if ncontinuoussamples[i] is None or ncontinuoussamples[i] ==0:
                pass
                #~ anaSigs.append(None)
            else :
                print 'i', i , ncontinuoussamples[i]
                anaSig = AnalogSignal(signal = zeros((ncontinuoussamples[i]) , dtype = 'f4'),
                                                    sampling_rate = float(slowChannelHeaders[i]['ADFreq']),
                                                    t_start = 0.,
                                                    )
                anaSig.name = slowChannelHeaders[i]['Name']
                anaSig.channel = slowChannelHeaders[i]['Channel']
                
                
                #~ anaSigs.append(anaSig)
                anaSigs[i] = anaSig
                
        # continuous : copy data chunks
        fid.seek(start)
        while fid.tell() !=-1 :
            dataBlockHeader = HeaderReader(fid , DataBlockHeader ).read_f(offset = None)
            if dataBlockHeader is None : break
            chan = dataBlockHeader['Channel']
            n1,n2 = dataBlockHeader['NumberOfWaveforms'] , dataBlockHeader['NumberOfWordsInWaveform']
            if n2 <0: break
            if dataBlockHeader['Type'] == 1:
                fid.seek(n1*n2*2 , 1)
                
            elif dataBlockHeader['Type'] == 4:
                pass
            
            elif dataBlockHeader['Type'] == 5:
                #~ print ''
                #~ for k,v in dataBlockHeader.iteritems(): print k,v
                if n2<= 0: 
                    print 'probable bug because NumberOfWordsInWaveform = ',n2, 'at fid.tell()', fid.tell()
                    continue
                data = fromstring( fid.read(n2*2) , dtype = 'i2').astype('f4')
                #range
                if globalHeader['Version'] ==100 or globalHeader['Version'] ==101 :
                    data = data*5000./(2048*slowChannelHeaders[chan]['Gain']*1000.)
                elif globalHeader['Version'] ==102 :
                    data = data*5000./(2048*slowChannelHeaders[chan]['Gain']*slowChannelHeaders[chan]['PreampGain'])
                elif globalHeader['Version'] >= 103:
                    data = data*globalHeader['SlowMaxMagnitudeMV']/(.5*(2**globalHeader['BitsPerSpikeSample'])*\
                                                        slowChannelHeaders[chan]['Gain']**slowChannelHeaders[chan]['PreampGain'])
                anaSigs[chan].signal[sampleposition[chan] : sampleposition[chan]+data.size] = data
                sampleposition[chan] += data.size
                if sampleposition[chan] ==0:
                    time = dataBlockHeader['UpperByteOf5ByteTimestamp']*2**32 + dataBlockHeader['TimeStamp']
                    time/=globalHeader['ADFrequency'] 
                    anaSigs[chan].t_start = time
            
            else :
                print 'probable bug because unkonwn type = ',dataBlockHeader['Type'], 'at fid.tell()', fid.tell()
                pass
                #~ print 'unkonwn type',dataBlockHeader['Type']
                #~ print '# tell 2#', fid.tell()
                break
                
        # add AnalogSignal to sgement
        for k,anaSig in anaSigs.iteritems() :
            if anaSig is not None:
                seg._analogsignals.append(anaSig)
                
        # add SpikeTrain to sgement
        for i in xrange(TSCounts.shape[0]):
            for j in xrange(5):
                spikeTr = spiketrains[i][j]
                if spikeTr is not None:
                    spikeTr.channel = i
                    spikeTr.unit = j
                    seg._spiketrains.append(spikeTr)

        
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



