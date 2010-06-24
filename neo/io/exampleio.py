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
    supported_objects  = [ Segment , AnalogSignal, SpikeTrain, Event, Epoch]
    # This class can return either a Block or a Segment
    # The first one is the default ( self.read )
    readable_objects    = [ Segment , AnalogSignal , SpikeTrain] 
    # This class is not able to write objects
    writeable_objects   = []

    has_header         = False
    is_streameable     = False
    
    # This is for GUI stuf : a definition for parameters when reading.
    read_params        = {
                        Segment : [
                                ('segment_duration' , { 'value' : 15., 
                                                'label' : 'Segment size (s.)' } ),
                                ('num_analogsignal' , { 'value' : 8,
                                                'label' : 'Number of recording points' } ),
                                ('num_spiketrain' , { 'value' : 3,
                                                'label' : 'Num of spiketrains' } ),

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
        Return a neo.Segment
        See read_segment for detail.
        """
        # the higher level of my IO is Segment so:
        return self.read_segment( **kargs)

    
    # write is not supported so I do not over class write from BaseIO

    
    # Segment reading is supported so I define this :
    def read_segment(self, 
                                        segment_duration = 15.,
                                        
                                        num_analogsignal = 4,
                                        num_spiketrain_by_channel = 3,
                                        
                                        ):
        """
        Return a fake Segment.
        
        The self.filename does not matter.
        
        In this IO read by default a Segment.
        
        This is just a example to be adapted to each ClassIO.
        In this case these 3 paramters are  taken in account because this function
        return a generated segment with fake AnalogSignal and fake SpikeTrain.
        
        segment_duration is the size in secend of the segment.
        num_analogsignal number of AnalogSignal in this segment
        num_spiketrain number of SpikeTrain in this segment
        
        """
        
        sampling_rate = 10000. #Hz
        t_start = -1.
        
        
        #time vector for generated signal
        t = arange(t_start, t_start+ segment_duration , 1./sampling_rate)
        
        # create an empty segment
        seg = Segment()
        
        # read nested analosignal
        for i in range(num_analogsignal):
            ana = self.read_analogsignal( channel = i ,segment_duration = segment_duration, t_start = t_start)
            seg._analogsignals += [ ana ]
        
        # read nested spiketrain
        for i in range(num_analogsignal):
            for j in range(num_spiketrain_by_channel):
                sptr = self.read_spiketrain(segment_duration = segment_duration, t_start = t_start , channel = i)
                seg._spiketrains += [ sptr ]
        
        
        # create event and epoch
        # note that they are not accessible directly
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
        
    
    def read_analogsignal(self , channel = 0,
                                                segment_duration = 15.,
                                                t_start = -1,
                                                ):
        """
        With this IO AnalogSignal can e acces directly with its channel number
        
        """
        sampling_rate = 10000.
        
        #time vector for generated signal
        t = arange(t_start, t_start+ segment_duration , 1./sampling_rate)
        
        # create analogsignal
        anasig = AnalogSignal()
        anasig.sampling_rate = sampling_rate
        anasig.t_start = t_start
        anasig.t_stop = t_start + segment_duration
        f1 = 3. # Hz
        anasig.signal = sin(2*pi*t*f1 + channel/5.*2*pi)+rand(t.size)
        anasig.channel = channel
        
        return anasig
        
        
        
        
        
    def read_spiketrain(self ,
                                                segment_duration = 15.,
                                                t_start = -1,
                                                channel = 0,
                                                ):
        """
        With this IO SpikeTrain can e acces directly with its channel number
        """
        # There are 2 possibles behaviour for a SpikeTrain
        # holding many Spike instance or directly holding spike times
        # we choose here the first : 

        num_spike_by_spiketrain = 40
        sampling_rate = 10000.
        
        #generate a fake spike shape (2d array if trodness >1)
        sig1 = -stats.nct.pdf(arange(11,60,4), 5,20)[::-1]/3.
        sig2 = stats.nct.pdf(arange(11,60,2), 5,20)
        sig = r_[ sig1 , sig2 ]
        
        
        basicshape = -sig/max(sig)
        basicshape = resample( basicshape , int(basicshape.size * sampling_rate / 10000. ) )
        wsize = basicshape.size
        
        # create a spiketrain
        spiketr = SpikeTrain()
        spiketr.sampling_rate = sampling_rate
        spiketr.t_start = t_start
        
        spiketr._spikes = [ ]
        spiketr.name = 'Neuron'
        spiketr.channel = channel
        for k in range( num_spike_by_spiketrain ):
                sp = Spike()
                sp.time = random.rand()*segment_duration+t_start
                sp.sampling_rate = sampling_rate
                factor = randn()/6+1 # nose factor in amplitude
                sp.waveform = basicshape * factor
                spiketr._spikes.append(sp)
        
        return spiketr



