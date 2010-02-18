# -*- coding: utf-8 -*-
"""

Classe for reading data from Plexion acquisition system (.plx)

Compatible from version 100 to 105.
Other versions are incontroled.


Supported : Read

@author :  luc estebanez ,sgarcia


"""






from baseio import BaseIO
from neo.core import *

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
    
    name               = 'NeuroExplorer'
    extensions          = [ 'nex' ]
    

    
    def __init__(self , filename = None) :
        """
        This class read a abf file.
        
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
        for k,v in globalHeader.iteritems(): print k,v
        
        seg.filedatetime = datetime.datetime(  globalHeader['Year'] , globalHeader['Month']  , globalHeader['Day'] ,
                    globalHeader['Hour'] , globalHeader['Minute'] , globalHeader['Second'] )
        
        print 'version' , globalHeader['Version']
        
        channelHeaders = { }
        # dsp channels header
        for i in xrange(globalHeader['NumDSPChannels']):
            print 'i',i
            channelHeader = HeaderReader(fid , ChannelHeader ).read_f(offset = None)
            channelHeader['Template'] = array(channelHeader['Template']).reshape((5,64))
            channelHeader['Boxes'] = array(channelHeader['Boxes']).reshape((5,2,4))
            channelHeaders[channelHeader['Channel']]=channelHeader
            
            #~ for k,v in channelHeader.iteritems(): print k,v
        print 'channelHeaders' , len(channelHeaders)
        print ''
        for k,v in channelHeaders.iteritems(): print k,v
        
        
        
        
        # event channel header
        eventHeaders = { }
        for i in xrange(globalHeader['NumEventChannels']):
            print ''
            print 'channel events',i
            eventHeader = HeaderReader(fid , EventHeader ).read_f(offset = None)
            for k,v in eventHeader.iteritems(): print k,v
            print 'ici' , eventHeader['Channel']
            eventHeaders[eventHeader['Channel']] = eventHeader
        print 'eventHeaders' , len(eventHeaders)
        print ''
        for k,v in eventHeaders.iteritems(): print k,v

            #~ for k,v in eventHeader.iteritems(): print k,v
        
        slowChannelHeaders = { }
        # slow channel header
        for i in xrange(globalHeader['NumSlowChannels']):
            #~ print 'i',i
            slowChannelHeader = HeaderReader(fid , SlowChannelHeader ).read_f(offset = None)
            slowChannelHeaders[slowChannelHeader['Channel']] = slowChannelHeader
            #~ for k,v in slowChannelHeader.iteritems(): print k,v
        print 'slowChannelHeaders' , len(slowChannelHeaders)
        print ''
        for k,v in slowChannelHeaders.iteritems(): print k,v
        
        
        ncontinuoussamples = zeros(len(slowChannelHeaders))
        sampleposition = zeros(len(slowChannelHeaders))
        
        # data block header
        start = fid.tell()
        while fid.tell() !=-1 :
            dataBlockHeader = HeaderReader(fid , DataBlockHeader ).read_f(offset = None)
            if dataBlockHeader is None : break
            chan = dataBlockHeader['Channel']
            time = dataBlockHeader['UpperByteOf5ByteTimestamp']*2**32 + dataBlockHeader['TimeStamp']
            time/=globalHeader['ADFrequency'] 
            
            
            if dataBlockHeader['Type'] == 1:
                #spike
                for k,v in dataBlockHeader.iteritems(): print k,v
                n1,n2 = dataBlockHeader['NumberOfWaveforms'] , dataBlockHeader['NumberOfWordsInWaveform']
                #~ print 'n&,n2', n1,n2
                data = fromstring( fid.read(n1*n2*2) , dtype = 'i2').reshape(n1,n2).astype('f')
                #range
                if globalHeader['Version'] <103:
                    data = data*3000./(2048*channelHeaders[chan]['Gain']*1000.)
                elif globalHeader['Version'] >=103 and globalHeader['Version'] <105:
                    data = data*globalHeader['SpikeMaxMagnitudeMV']/(.5*2.**(globalHeader['BitsPerSpikeSample'])*1000.)
                elif globalHeader['Version'] >105:
                    data = data*globalHeader['SpikeMaxMagnitudeMV']/(.5*2.**(globalHeader['BitsPerSpikeSample'])*globalHeader['SpikePreAmpGain'])
            
            if dataBlockHeader['Type'] ==4:
                #event
                #~ print ''
                for k,v in dataBlockHeader.iteritems(): print k,v
                ev = Event(time = time)
                ev.name = eventHeaders[chan]['Name']
                seg._events.append(ev)


            if dataBlockHeader['Type'] == 5:
                #continuous
                #~ print ''
                #~ for k,v in dataBlockHeader.iteritems(): print k,v
                #~ data = fromstring( fid.read(dataBlockHeader['NumberOfWordsInWaveform']*2) , dtype = 'i2').astype('f')
                fid.seek(fid.tell()+dataBlockHeader['NumberOfWordsInWaveform']*2)
                ncontinuoussamples[chan] += dataBlockHeader['NumberOfWordsInWaveform']
        
        # continuous
        anaSigs = [ ]
        for i in range(ncontinuoussamples.size):
            if ncontinuoussamples[i] is None:
                anaSigs.append(None)
            else :
                anaSig = AnalogSignal(signal = zeros((ncontinuoussamples[i]) , dtype = 'f4'),
                                                    freq = float(slowChannelHeaders[i]['ADFreq']),
                                                    t_start = 0,
                                                    )
                anaSig.name = slowChannelHeaders[i]['Name']
                anaSigs.append(anaSig)
                
        fid.seek(start)
        while fid.tell() !=-1 :
            dataBlockHeader = HeaderReader(fid , DataBlockHeader ).read_f(offset = None)
            if dataBlockHeader is None : break
            chan = dataBlockHeader['Channel']
            if dataBlockHeader['Type'] == 1:
                n1,n2 = dataBlockHeader['NumberOfWaveforms'] , dataBlockHeader['NumberOfWordsInWaveform']
                fid.seek(fid.tell()+n1*n2*2)
                
            if dataBlockHeader['Type'] == 4:
                pass
            
            if dataBlockHeader['Type'] == 5:
                #~ print ''
                #~ for k,v in dataBlockHeader.iteritems(): print k,v
                data = fromstring( fid.read(dataBlockHeader['NumberOfWordsInWaveform']*2) , dtype = 'i2').astype('f4')
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
                
                
                ncontinuoussamples[chan] += dataBlockHeader['NumberOfWordsInWaveform']
                
                #~ print data.shape
        
        for anaSig in anaSigs :
            if anaSig is not None:
                seg._analogsignals.append(anaSig)
        
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
    
    # all version
    ('Padding','46s'),
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
    ('Comment','128s'),
    ('Padding','11i'),
    ]

EventHeader = [
    ('Name' , '32s'),
    ('Channel','i'),
    ('Comment' , '128s'),
    ('Padding','33i'),
    ]


SlowChannelHeader = [
    ('Name' , '32s'),
    ('Channel','i'),
    ('ADFreq','i'),
    ('Gain','i'),
    ('Enabled','i'),
    ('PreamGain','i'),
    #version 104
    ('SpikeChannel','i'),
    ('Comment','128s'),
    ('Padding','28i'),
    ]

DataBlockHeader = [
    ('Type','h'),
    ('UpperByteOf5ByteTimestamp','h'),
    ('TimeStamp','i'),
    ('Channel','h'),
    ('Unit','h'),
    ('NumberOfWaveforms','h'),
    ('NumberOfWordsInWaveform','h'),
    ]


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



