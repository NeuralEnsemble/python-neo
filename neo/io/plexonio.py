# -*- coding: utf-8 -*-
"""
Class for reading data from Plexion acquisition system (.plx)

Compatible with versions 100 to 106.
Other versions have not been tested.

This IO is developed thanks to the header file downloadable from:
http://www.plexon.com/downloads.html

Supported: Read

Author: sgarcia

"""

import datetime
import struct
import os

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Segment, AnalogSignal, SpikeTrain, EpochArray, EventArray
from neo.io.tools import iteritems


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

    is_readable = True
    is_writable = False

    supported_objects = [Segment, AnalogSignal, SpikeTrain, EventArray, EpochArray]
    readable_objects = [Segment]
    writeable_objects = []

    has_header = False
    is_streameable = False

    # This is for GUI stuff: a definition for parameters when reading.
    read_params = {
        Segment:  [
            ('load_spike_waveform', {'value': False}),
            ]
        }
    write_params = None

    name = 'Plexon'
    extensions = ['plx']

    mode = 'file'


    def __init__(self, filename=None):
        """
        Arguments:
            filename : the filename

        """
        BaseIO.__init__(self, filename)

    def read_segment(self, lazy=False, cascade=True, load_spike_waveform=True):
        """
        Read in a segment.

        Arguments:
            load_spike_waveform : load or not waveform of spikes (default True)

        """

        fid = open(self.filename, 'rb')
        globalHeader = HeaderReader(fid, GlobalHeader).read_f(offset=0)

        # metadatas
        seg = Segment()
        seg.rec_datetime = datetime.datetime(
            globalHeader.pop('Year'),
            globalHeader.pop('Month'),
            globalHeader.pop('Day'),
            globalHeader.pop('Hour'),
            globalHeader.pop('Minute'),
            globalHeader.pop('Second')
        )
        seg.file_origin = os.path.basename(self.filename)

        for key, val in globalHeader.iteritems():
            seg.annotate(**{key: val})

        if not cascade:
            return seg

        ## Step 1 : read headers
        # dsp channels header = spikes and waveforms
        dspChannelHeaders = {}
        maxunit = 0
        maxchan = 0
        for _ in range(globalHeader['NumDSPChannels']):
            # channel is 1 based
            channelHeader = HeaderReader(fid, ChannelHeader).read_f(offset=None)
            channelHeader['Template'] = np.array(channelHeader['Template']).reshape((5,64))
            channelHeader['Boxes'] = np.array(channelHeader['Boxes']).reshape((5,2,4))
            dspChannelHeaders[channelHeader['Channel']] = channelHeader
            maxunit = max(channelHeader['NUnits'], maxunit)
            maxchan = max(channelHeader['Channel'], maxchan)

        # event channel header
        eventHeaders = { }
        for _ in range(globalHeader['NumEventChannels']):
            eventHeader = HeaderReader(fid, EventHeader).read_f(offset=None)
            eventHeaders[eventHeader['Channel']] = eventHeader

        # slow channel header = signal
        slowChannelHeaders = {}
        for _ in range(globalHeader['NumSlowChannels']):
            slowChannelHeader = HeaderReader(fid, SlowChannelHeader).read_f(offset=None)
            slowChannelHeaders[slowChannelHeader['Channel']] = slowChannelHeader

        ## Step 2 : a first loop for counting size
        # signal
        nb_samples = np.zeros(len(slowChannelHeaders))
        sample_positions = np.zeros(len(slowChannelHeaders))
        t_starts = np.zeros(len(slowChannelHeaders), dtype='f')

        #spiketimes and waveform
        nb_spikes = np.zeros((maxchan+1, maxunit+1) ,dtype='i')
        wf_sizes = np.zeros((maxchan+1, maxunit+1, 2) ,dtype='i')

        # eventarrays
        nb_events = { }
        #maxstrsizeperchannel = { }
        for chan, h in iteritems(eventHeaders):
            nb_events[chan] = 0
            #maxstrsizeperchannel[chan] = 0

        start = fid.tell()
        while fid.tell() !=-1 :
            # read block header
            dataBlockHeader = HeaderReader(fid , DataBlockHeader ).read_f(offset = None)
            if dataBlockHeader is None : break
            chan = dataBlockHeader['Channel']
            unit = dataBlockHeader['Unit']
            n1,n2 = dataBlockHeader['NumberOfWaveforms'] , dataBlockHeader['NumberOfWordsInWaveform']
            time = (dataBlockHeader['UpperByteOf5ByteTimestamp']*2.**32 +
                    dataBlockHeader['TimeStamp'])

            if dataBlockHeader['Type'] == 1:
                nb_spikes[chan,unit] +=1
                wf_sizes[chan,unit,:] = [n1,n2]
                fid.seek(n1*n2*2,1)
            elif dataBlockHeader['Type'] ==4:
                #event
                nb_events[chan] += 1
            elif dataBlockHeader['Type'] == 5:
                #continuous signal
                fid.seek(n2*2, 1)
                if n2> 0:
                    nb_samples[chan] += n2
                if nb_samples[chan] ==0:
                    t_starts[chan] = time
                    

        ## Step 3: allocating memory and 2 loop for reading if not lazy
        if not lazy:
            # allocating mem for signal
            sigarrays = { }
            for chan, h in iteritems(slowChannelHeaders):
                sigarrays[chan] = np.zeros(nb_samples[chan])
                
            # allocating mem for SpikeTrain
            stimearrays = np.zeros((maxchan+1, maxunit+1) ,dtype=object)
            swfarrays = np.zeros((maxchan+1, maxunit+1) ,dtype=object)
            for (chan, unit), _ in np.ndenumerate(nb_spikes):
                stimearrays[chan,unit] = np.zeros(nb_spikes[chan,unit], dtype = 'f')
                if load_spike_waveform:
                    n1,n2 = wf_sizes[chan, unit,:]
                    swfarrays[chan, unit] = np.zeros( (nb_spikes[chan, unit], n1, n2 ) , dtype = 'f4' )
            pos_spikes = np.zeros(nb_spikes.shape, dtype = 'i')
                    
            # allocating mem for event
            eventpositions = { }
            evarrays = { }
            for chan, nb in iteritems(nb_events):
                evarrays[chan] = {
                    'times': np.zeros(nb, dtype='f'),
                    'labels': np.zeros(nb, dtype='S4')
                }
                eventpositions[chan]=0 
                
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
                    pos = pos_spikes[chan,unit]
                    stimearrays[chan, unit][pos] = time
                    if load_spike_waveform and n1*n2 != 0 :
                        swfarrays[chan,unit][pos,:,:] = np.fromstring( fid.read(n1*n2*2) , dtype = 'i2').reshape(n1,n2).astype('f4')
                    else:
                        fid.seek(n1*n2*2,1)
                    pos_spikes[chan,unit] +=1
                
                elif dataBlockHeader['Type'] == 4:
                    # event
                    pos = eventpositions[chan]
                    evarrays[chan]['times'][pos] = time
                    evarrays[chan]['labels'][pos] = dataBlockHeader['Unit']
                    eventpositions[chan]+= 1

                elif dataBlockHeader['Type'] == 5:
                    #signal
                    data = np.fromstring( fid.read(n2*2) , dtype = 'i2').astype('f4')
                    sigarrays[chan][sample_positions[chan] : sample_positions[chan]+data.size] = data
                    sample_positions[chan] += data.size


        ## Step 4: create neo object
        for chan, h in iteritems(eventHeaders):
            if lazy:
                times = []
                labels = None
            else:
                times = evarrays[chan]['times']
                labels = evarrays[chan]['labels']
            ea = EventArray(
                times*pq.s,
                labels=labels,
                channel_name=eventHeaders[chan]['Name'],
                channel_index=chan
            )
            if lazy:
                ea.lazy_shape = nb_events[chan]
            seg.eventarrays.append(ea)

            
        for chan, h in iteritems(slowChannelHeaders):
            if lazy:
                signal = [ ]
            else:
                if globalHeader['Version'] ==100 or globalHeader['Version'] ==101 :
                    gain = 5000./(2048*slowChannelHeaders[chan]['Gain']*1000.)
                elif globalHeader['Version'] ==102 :
                    gain = 5000./(2048*slowChannelHeaders[chan]['Gain']*slowChannelHeaders[chan]['PreampGain'])
                elif globalHeader['Version'] >= 103:
                    gain = globalHeader['SlowMaxMagnitudeMV']/(.5*(2**globalHeader['BitsPerSpikeSample'])*\
                                                        slowChannelHeaders[chan]['Gain']*slowChannelHeaders[chan]['PreampGain'])
                signal = sigarrays[chan]*gain
            anasig =  AnalogSignal(signal*pq.V,
                sampling_rate = float(slowChannelHeaders[chan]['ADFreq'])*pq.Hz,
                t_start = t_starts[chan]*pq.s,
                channel_index = slowChannelHeaders[chan]['Channel'],
                channel_name = slowChannelHeaders[chan]['Name'],
            )
            if lazy:
                anasig.lazy_shape = nb_samples[chan]
            seg.analogsignals.append(anasig)
            
        for (chan, unit), value in np.ndenumerate(nb_spikes):
            if nb_spikes[chan, unit] == 0: continue
            if lazy:
                times = [ ]
                waveforms = None
                t_stop = 0
            else:
                times = stimearrays[chan,unit]
                t_stop = times.max()
                if load_spike_waveform:
                    if globalHeader['Version'] <103:
                        gain = 3000./(2048*dspChannelHeaders[chan]['Gain']*1000.)
                    elif globalHeader['Version'] >=103 and globalHeader['Version'] <105:
                        gain = globalHeader['SpikeMaxMagnitudeMV']/(.5*2.**(globalHeader['BitsPerSpikeSample'])*1000.)
                    elif globalHeader['Version'] >105:
                        gain = globalHeader['SpikeMaxMagnitudeMV']/(.5*2.**(globalHeader['BitsPerSpikeSample'])*globalHeader['SpikePreAmpGain'])                    
                    waveforms = swfarrays[chan, unit] * gain * pq.V
                else:
                    waveforms = None
            sptr = SpikeTrain(
                times,
                units='s', 
                t_stop=t_stop*pq.s,
                waveforms=waveforms
            )
            sptr.annotate(unit_name = dspChannelHeaders[chan]['Name'])
            sptr.annotate(channel_index = chan)
            for key, val in dspChannelHeaders[chan].iteritems():
                sptr.annotate(**{key: val})

            if lazy:
                sptr.lazy_shape = nb_spikes[chan,unit]
            seg.spiketrains.append(sptr)

        seg.create_many_to_one_relationship()
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

    def __init__(self, fid, description):
        self.fid = fid
        self.description = description

    def read_f(self, offset=None):
        if offset is not None:
            self.fid.seek(offset)
        d = {}
        for key, fmt in self.description:
            buf = self.fid.read(struct.calcsize(fmt))
            if len(buf) != struct.calcsize(fmt): return None
            val = list(struct.unpack(fmt , buf))
            for i, ival in enumerate(val):
                if hasattr(ival, 'replace'):
                    val[i] = ival.replace('\x00','')
            if len(val) == 1:
                val = val[0]
            d[key] = val
        return d



