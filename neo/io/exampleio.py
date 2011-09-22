# encoding: utf-8
"""
Class for "reading" fake data from an imaginary file.

For the user, it generates a :class:`Segment` or a :class:`Block` with a
sinusoidal :class:`AnalogSignal`, a :class:`SpikeTrain` and an
:class:`EventArray`.

For a developer, it is just an example showing guidelines for someone who wants
to develop a new IO module.

Depends on: scipy

Supported: Read

Author: sgarcia

"""
from __future__ import absolute_import

# I need to subclass BaseIO
from .baseio import BaseIO

# to import from core
from ..core import Block, Segment, AnalogSignal, SpikeTrain, EventArray

# some tools to finalize the hierachy
from .tools import create_many_to_one_relationship

# note neo.core needs only numpy and quantities
import numpy as np
import quantities as pq

# but my specific IO can depend on many other packages
from numpy import pi, newaxis
import datetime
try:
    have_scipy = True
    from scipy import stats
    from scipy import randn, rand
    from scipy.signal import resample
except ImportError:
    have_scipy = False


# I need to subclass BaseIO
class ExampleIO(BaseIO):
    """
    Class for "reading" fake data from an imaginary file.

    For the user, it generates a :class:`Segment` or a :class:`Block` with a
    sinusoidal :class:`AnalogSignal`, a :class:`SpikeTrain` and an
    :class:`EventArray`.

    For a developer, it is just an example showing guidelines for someone who wants
    to develop a new IO module.
  
    Two rules for developers:
      * Respect the Neo IO API (:ref:`neo_io_API`)
      * Follow :ref:`io_guiline`
    
    Usage:
        >>> from neo import io
        >>> r = io.ExampleIO(filename='itisafake.nof')
        >>> seg = r.read_segment(lazy=False, cascade=True)
        >>> print(seg.analogsignals)
        >>> print(seg.spiketrains)
        >>> print(seg.eventarrays)
        >>> anasig = r.read_analogsignal(lazy=True, cascade=False)
        >>> print(anasig._data_description)
        >>> anasig = r.read_analogsignal(lazy=False, cascade=False)
        
    """
    
    is_readable = True # This class can only read data
    is_writable = False # write is not supported
    
    # This class is able to directly or indirectly handle the following objects
    # You can notice that this greatly simplifies the full Neo object hierarchy
    supported_objects  = [ Segment , AnalogSignal, SpikeTrain, EventArray ]
    
    # This class can return either a Block or a Segment
    # The first one is the default ( self.read )
    readable_objects    = [ Segment , AnalogSignal, SpikeTrain ]
    # This class is not able to write objects
    writeable_objects   = [ ]

    has_header         = False
    is_streameable     = False
    
    # This is for GUI stuff : a definition for parameters when reading.
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
    
    # do not supported write so no GUI stuff
    write_params       = None
    
    name               = 'example'
    
    extensions          = [ 'nof' ]
    
    # mode can be 'file' or 'dir' or 'fake' or 'database'
    # the main case is 'file' but some reader are base on a directory or a database
    # this info is for GUI stuff also
    mode = 'fake' 
    

    
    def __init__(self , filename = None) :
        """
        
        
        Arguments:
            filename : the filename
            
        Note:
            - filename is here just for exampe because it will not be take in account
            - if mode=='dir' the argument should be dirname (See TdtIO)

        """
        BaseIO.__init__(self)
        self.filename = filename


    # Segment reading is supported so I define this :
    def read_segment(self, 
                     # the 2 first keyword arguments are imposed by neo.io API
                     lazy = False,
                     cascade = True,                   
                     # all following arguments are decied by this IO and are free
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
        
        Parameters:
            segment_duration :is the size in secend of the segment.
            num_analogsignal : number of AnalogSignal in this segment
            num_spiketrain : number of SpikeTrain in this segment
        
        """
        
        sampling_rate = 10000. #Hz
        t_start = -1.
        
        
        #time vector for generated signal
        timevect = np.arange(t_start, t_start+ segment_duration , 1./sampling_rate)
        
        # create an empty segment
        seg = Segment( name = 'it is a seg from exampleio')
        
        if cascade:
            # read nested analosignal
            for i in range(num_analogsignal):
                ana = self.read_analogsignal( lazy = lazy , cascade = cascade ,
                                            channel_index = i ,segment_duration = segment_duration, t_start = t_start)
                seg.analogsignals += [ ana ]
            
            # read nested spiketrain
            for i in range(num_analogsignal):
                for j in range(num_spiketrain_by_channel):
                    sptr = self.read_spiketrain(lazy = lazy , cascade = cascade ,
                                                            segment_duration = segment_duration, t_start = t_start , channel_index = i)
                    seg.spiketrains += [ sptr ]
        
            
            # create an EventArray that mimic triggers.
            # note that ExampleIO  do not allow to acess directly to EventArray
            # for that you need read_segment(cascade = True)
            eva = EventArray()
            if lazy:
                # in lazy case no data are readed
                # eva is empty
                pass
            else:
                # otherwise it really contain data
                n = 1000
                
                # neo.io support quantities my vector use second for unit
                eva.times = timevect[(rand(n)*timevect.size).astype('i')]* pq.s
                # all duration are the same
                eva.durations = np.ones(n)*500*pq.ms
                # label
                l = [ ]
                for i in range(n):
                    if rand()>.6: l.append( 'TriggerA' )
                    else : l.append( 'TriggerB' )
                eva.labels = np.array( l )
                
            seg.eventarrays += [ eva ]
        
        create_many_to_one_relationship(seg)
        return seg
        
    
    def read_analogsignal(self ,
                          # the 2 first key arguments are imposed by neo.io API
                          lazy = False,
                          cascade = True,
                          channel_index = 0,
                          segment_duration = 15.,
                          t_start = -1,
                          ):
        """
        With this IO AnalogSignal can e acces directly with its channel number
        
        """
        sr = 10000.
        sinus_freq = 3. # Hz
        tvect = np.arange(t_start, t_start+ segment_duration , 1./sr)
        
        if lazy:
            anasig = AnalogSignal([ ], units = 'V', sampling_rate=sr*pq.Hz, t_start=t_start*pq.s)
            anasig._data_description = {'shape' : tvect.shape}
        else:
            #time vector for generated signal

            # create analogsignal (sinus of 3 Hz)
            sig = np.sin(2*pi*tvect*sinus_freq + channel_index/5.*2*pi)+rand(tvect.size)
            anasig = AnalogSignal(sig, units= 'V' ,  sampling_rate = sr * pq.Hz , t_start = t_start*pq.s)
        
        anasig._annotations['channel_index'] = channel_index
        anasig._annotations['info'] = 'it is a sinus of %f Hz' %sinus_freq
        
        return anasig
        
        
        
        
        
    def read_spiketrain(self ,
                                            # the 2 first key arguments are imposed by neo.io API
                                            lazy = False,
                                            cascade = True,
        
                                                segment_duration = 15.,
                                                t_start = -1,
                                                channel_index = 0,
                                                ):
        """
        With this IO SpikeTrain can e acces directly with its channel number
        """
        # There are 2 possibles behaviour for a SpikeTrain
        # holding many Spike instance or directly holding spike times
        # we choose here the first : 

        num_spike_by_spiketrain = 40
        sr = 10000.
        
        if lazy:
            times = [ ]
        else:
            times = rand(num_spike_by_spiketrain)*segment_duration+t_start
        
        # create a spiketrain
        spiketr = SpikeTrain(times, t_start = t_start*pq.s, t_stop = (t_start+segment_duration)*pq.s ,
                                            units = pq.s, 
                                            name = 'it is a spiketrain from exampleio',
                                            )
        
        if lazy:
            spiketr._data_description = {
                                                        'times_shape' : num_spike_by_spiketrain,
                                                        }
        
        # ours spiketrains also hold the waveforms:
        
        # 1 generate a fake spike shape (2d array if trodness >1)
        w1 = -stats.nct.pdf(np.arange(11,60,4), 5,20)[::-1]/3.
        w2 = stats.nct.pdf(np.arange(11,60,2), 5,20)
        w = np.r_[ w1 , w2 ]
        w = -w/max(w)
        
        # in the neo API the waveforms attr is 3 D in case tetrode
        # in our case it is mono electrode so dim 1 is size 1
        waveforms  = np.tile( w[newaxis,newaxis,:], ( num_spike_by_spiketrain ,1, 1) )
        waveforms *=  randn(*waveforms.shape)/6+1
        spiketr.waveforms = waveforms
        spiketr.sampling_rate = sr * pq.Hz
        spiketr.left_sweep = 1.5* pq.s
        
        spiketr._annotations['channel_index'] = channel_index
        
        return spiketr




