# -*- coding: utf-8 -*-
"""
Classe for fake reading data in a no file.

For the user, it generate a `Segment` or a `Block` with `AnalogSignal` sinusoidale + `SpikeTrain` + `Event`

For a developper, it is just a example for guidelines who want to develop a new IO.

Supported : Read

@author : sgarcia


"""

# I need to subclass BaseIO
from baseio import BaseIO
# to import : Block, Segment, AnalogSignal, SpikeTrain, SpikeTrainList
#from neo.core import *

from ..core import *

import datetime

# So bad :
from numpy import *
from scipy import stats
from scipy import randn, rand
from scipy.signal import resample




# I need to subclass BaseIO
class ExampleIO(BaseIO):
    """
    Class for reading/writing data in a fake file.
    
    **For developpers**
    
    If you start a new IO class :
        - Copy/paste and modify this class.
        - Think what objects your IO will support
        - Think what objects your IO will read or write.
        - Implement all read_XXX and write_XXX methods

    If you have a problem just mail me or ask the list.

    **Guidelines**
        - Each IO implementation of BaseFile can also add attributs (fields) freely to all object.
        - Each IO implementation of BaseFile should come with tipics files exemple in neo/test/unitest/io/datafiles.
        - Each IO implementation of BaseFile should come with its documentation.
        - Each IO implementation of BaseFile should come with its unitest neo/test/unitest/io.
        
    **Usage**
     
    **Example**
    
    """
    
    is_readable        = True # This a only reading class
    is_writable        = False
    #This class is able directly or inderectly this kind of objects
    supported_objects  = [Block, Segment , RecordingPoint , AnalogSignal, SpikeTrain, Event, Epoch]
    # This class can return either a Block or a Segment
    # The first one is the default ( self.read )
    readable_objects    = [Block, Segment]
    # This class is not able to write objects
    writeable_objects   = []

    has_header         = False
    is_streameable     = False
    
    # This is for GUI stuf : a definition for parameters when reading.
    read_params        = {
                        Block : [
                                ('segmentduration' , { 'value' : 15., 
                                                'label' : 'Segment size (s.)' } ),
                                ('num_segment' , { 'value' : 5,
                                              'label' : 'Segment number' } ),
                                ('num_recordingpoint' , { 'value' : 8,
                                                'label' : 'Number of recording points' } ),
                                ('num_spiketrainbyrecordingpoint' , { 'value' : 3,
                                                'label' : 'Num of spiketrain by recording points' } ),
                                ('trodness' , { 'value' : 4,
                                                'label' : 'trdness (1= normal 2=stereotrode   4=tetrode)' } ),
                                
                                ('spike_amplitude' , { 'value' : .8,
                                                'label' : 'Amplitude of spikes' } ),
                                ('sinus_amplitude' , { 'value' : 1.,
                                                'label' : 'Amplitude of sinus' } ),
                                ('randnoise_amplitude' , { 'value' : .4,
                                                'label' : 'Rand noise' } ),
                                
                                ],
                        Segment : [
                                ('segmentduration' , { 'value' : 15., 
                                                'label' : 'Segment size (s.)' } ),
                                ('num_recordingpoint' , { 'value' : 8,
                                                'label' : 'Number of recording points' } ),
                                ('num_spiketrainbyrecordingpoint' , { 'value' : 3,
                                                'label' : 'Num of spiketrain by recording points' } ),
                                ('trodness' , { 'value' : 4,
                                                'label' : 'trdness (1= normal 2=stereotrode   4=tetrode)' } ),

                                ('spike_amplitude' , { 'value' : .8,
                                                'label' : 'Amplitude of spikes' } ),
                                ('sinus_amplitude' , { 'value' : 1.,
                                                'label' : 'Amplitude of sinus' } ),
                                ('randnoise_amplitude' , { 'value' : .4,
                                                'label' : 'Rand noise' } ),
                                    ],
                        }
    
    # do not supported write so no GUI stuf
    write_params       = None
    
    name               = 'example'
    extensions          = [ 'fak' ]
    
    filemode = False
    

    
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
        Return a neo.Block
        See read_block for detail.
        """
        return self.read_block( **kargs)
    
    # write is not supported so I do not over class write from BaseIO

    
    
    # Block reading is supported so I define this :
    def read_block(self , 
                                        num_segment = 5,
                                        
                                        segmentduration = 15.,
                                        
                                        num_recordingpoint = 8,
                                        num_spiketrainbyrecordingpoint = 3,
                                        
                                        trodness = 4,
                                        num_spike_by_spiketrain = 30,
                                        
                                        spike_amplitude = .8,
                                        sinus_amplitude = 1.,
                                        randnoise_amplitude = 0.4,                             
                                        
                        ) :
        """
        Return a fake Block.
        
        **Arguments**
        
        num_segment : the number of segment in the file
        
        segmentduration : duration in second for each segment
        
        num_recordingpoint : number of recording point in one segment
                                one AnalogSignal is return for one RecordingPoint
                                
        num_spiketrainbyrecordingpoint : number of SpikeTrain for one RecordingPoint
        
        """
        
        if num_recordingpoint%trodness != 0:
            num_recordingpoint = (num_recordingpoint/trodness) * trodness
        
        blck = Block()
        blck.name = 'example block'
        blck.datetime = datetime.datetime.now()
        self.props = rand(num_spiketrainbyrecordingpoint, trodness)# this is for spikes
        for i in range(num_segment) :
            # read a segment in the fake file
            # id_segment is just a example it is not taken in account
            seg = self.read_segment(id_segment = i,
                                        segmentduration = segmentduration,
                                        num_recordingpoint = num_recordingpoint,
                                        num_spiketrainbyrecordingpoint = num_spiketrainbyrecordingpoint,
                                        trodness = trodness,
                                        num_spike_by_spiketrain = num_spike_by_spiketrain, 
                                        
                                        spike_amplitude = spike_amplitude,
                                        sinus_amplitude = sinus_amplitude,
                                        randnoise_amplitude = randnoise_amplitude,                                               
                                        )
            seg.name = 'segment %d' % i 
            seg.datetime = datetime.datetime.now()
            # Add seg to blck instance
            blck._segments.append( seg )
            
        
        # create recording point
        for i in range(num_recordingpoint):
            rp = RecordingPoint()
            rp.name = 'electrode %i' % i
            rp.group = int(i/trodness)+1
            rp.trodness = trodness
            rp.channel = i
            blck._recordingpoints.append(rp)
            
        # associate analogsignal of same recording point
        for i in range(num_recordingpoint):
            for seg in blck._segments:
                for ana in seg._analogsignals:
                    if ana.channel == blck._recordingpoints[i].channel:
                        blck._recordingpoints[i]._analogsignals.append( ana)
        
        # associate spiketrain of same recording point : neuron
        for i in range(num_recordingpoint/trodness):
            for j in range(num_spiketrainbyrecordingpoint):
                neu = Neuron(name = 'Neuron %d of recPoint %d' %( j , i*trodness) )
                blck._neurons.append( neu )
                for seg in blck._segments:
                    for sptr in seg._spiketrains :
                        if sptr.name == neu.name :
                            neu._spiketrains.append( sptr)
                            blck._recordingpoints[sptr.channel]._spiketrains.append( sptr )
        
        return blck
        
    
    # Segment reading is supported so I define this :
    def read_segment(self, 
                                        filename = '',
                                        
                                        num_segment = 12,
                                        id_segment = 0,
                                        name_segment = 'test',
                                        
                                        segmentduration = 15.,
                                        
                                        num_recordingpoint = 4,
                                        num_spiketrainbyrecordingpoint = 3,
                                        
                                        trodness = 4,
                                        num_spike_by_spiketrain = 30,
                                        
                                        spike_amplitude = .8,
                                        sinus_amplitude = 1.,
                                        randnoise_amplitude = 0.4,
                                        
                                        ):
        """
        Return a fake Segment.
        
        The filename does not matter.
        
        In this IO read by default a Block.
        Segment is readable so it is a nested read.
        So we need to define a num_segment, or a id_segment or a name_segment.
        
        This is just a example to be adapted to each ClassIO.
        In this case these 3 paramters are not taken in account because this function
        return a generated segment with fake AnalogSignal and fake SpikeTrain.
        
        segmentduration is the size in secend of the segment.
        
        In this example the segment is supposed to return one AnalogSignal for
        one RecordingPoint and some SpikeTrain for one RecordingPoint.
        This is a typical example for an extra cellular recording.
        This is controled by :
        num_recordingpoint
        num_spiketrainbyrecordingpoint
        trodness ( 4 = tetrode groups, 1 = monoelectrode groups )
        
        """
        
        sampling_rate = 10000. #Hz
        t_start = -1.
        
        #~ spike_amplitude = 1
        #~ sinus_amplitude = 0
        #~ randnoise_amplitude = 0.2
        #randnoise_amplitude = 0.
        
        
        
        #time vector for generated signal
        t = arange(t_start, t_start+ segmentduration , 1./sampling_rate)
        
        # create an empty segment
        seg = Segment()
        

        
        # create some SpikeTrain :

        #generate a fake spike shape (2d array if trodness >1)
        sig1 = -stats.nct.pdf(arange(11,60,4), 5,20)[::-1]/3.
        sig2 = stats.nct.pdf(arange(11,60,2), 5,20)
        sig = r_[ sig1 , sig2 ]
        basicshape = -sig/max(sig)
        basicshape = resample( basicshape , int(basicshape.size * sampling_rate / 10000. ) )
        wsize = basicshape.size

        for j in range(num_spiketrainbyrecordingpoint):
            
            # basic shape duplicate on each trodness electrode with a random factor
            
            
            for i in range(num_recordingpoint/trodness):
                #~ props = rand(num_spiketrainbyrecordingpoint, trodness)
                props = self.props
                
                
                spikeshape = empty( (0, basicshape.size, ))
                for k in range(trodness):
                    spikeshape = concatenate( (spikeshape, basicshape[newaxis,:]*props[i,k]) , axis = 0)
                
                spiketr = SpikeTrain()
                spiketr.sampling_rate = sampling_rate
                
                # There are 2 possibles behaviour for a SpikeTrain
                # holding many Spike instance or directly holding spike times
                # we choose here the first : 
                
                spiketr._spikes = [ ]
                spiketr.name = 'Neuron %d of recPoint %d' %( j , i*trodness)
                spiketr.channel = i*trodness
                for k in range( num_spike_by_spiketrain ):
                        sp = Spike()
                        sp.time = random.rand()*segmentduration+t_start
                        sp.sampling_rate = sampling_rate
                        factor = randn()/6+1 # nose factor in amplitude
                        sp.waveform = spikeshape * factor * spike_amplitude
                        spiketr._spikes.append(sp)
                
                # Add spiketr to seg instance
                seg._spiketrains.append( spiketr )
        
        
        # create some AnalogSignal :
        for i in range(num_recordingpoint):
            anasig = AnalogSignal()
            anasig.sampling_rate = sampling_rate
            anasig.t_start = t_start
            anasig.t_stop = t_start + segmentduration
            
            sig = zeros(t.shape, 'f')
            for s in range(1):
                # choose random freq between 20 and 80 for my sinus signal :
                #f1 = random.rand()*80+20.
                f1 = linspace(random.rand()*60+20. , random.rand()*60+20., t.size)
                # choose a random freq for modulation between .5 and 2
                
                #f2 = random.rand()*1.5+.5
                f2 = linspace(random.rand()*1.+.1 , random.rand()*1.+.1, t.size)
                sig1 = sin(2*pi*t*f1) * sin(pi*t*f2+random.rand()*pi)**2
                sig1[t<0] = 0.
                sig += sig1*sinus_amplitude
                
            anasig.signal = (random.rand(t.size)-.5)*randnoise_amplitude + sig
            anasig.channel = i
            anasig.name = 'signal on channel %d'%i
            
            for j in range(num_spiketrainbyrecordingpoint):
                #~ sptr = seg._spiketrains[ int(num_recordingpoint/trodness)+j]
                sptr = seg._spiketrains[ j ]
                for sp in sptr._spikes:
                    waveform = sp.waveform[  i % trodness,:  ]
                    pos = digitize( [sp.time] , t )
                    pos = pos[0]-wsize/2
                    if pos>=anasig.signal.size-wsize :
                        pos = anasig.signal.size-wsize-1
                    if pos<0 :
                        pos =0
                    anasig.signal[pos:pos+wsize] +=  waveform
                        
            
            # theses 2 following fields are optionals and specifics from my IO :
            anasig.unit = 'mV'
            anasig.label = 'fantastic signal %i' % i
            
            # Add anasig to seg instance
            seg._analogsignals.append( anasig )
        
        # create event and epoch
        n_event = 3
        n_epoch = 1
        for i in range(n_event):
            ev = Event( time = t[int(random.rand()*t.size)] )
            seg._events.append(ev)
        
        for i in range(n_epoch):
            time = t[int(random.rand()*t.size/2)] 
            ep = Epoch( time = time,
                                duration= time+1.,
                                )
            seg._epochs.append(ep)
        
        return seg


