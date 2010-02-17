# -*- coding: utf-8 -*-
"""

Classe for fake reading data in a no file.

For the user, it generate a `Segment` or a `Block` with `AnalogSignal` sinusoidale + `SpikeTrain` + `Event`

For a developper, it is just a example for guidelines who want to develop a new IO.

Supported : Read

@author : sgarcia


"""




"""

For developpers :

If you start a new IO class :
1 - copy/paste and modify this class.
2 - Think what objects my IO support
3 - Think what objects my IO can read or write.
4 - Implement all read_XXX and write_XXX methods

If you have a problem just mail me or ask the list.



    ** Guidelines **
        Each IO implementation of BaseFile can also add attributs (fields) freely to all object.
        Each IO implementation of BaseFile should come with tipics files exemple in neo/test/unitest/io/datafiles.
        Each IO implementation of BaseFile should come with its documentation.
        Each IO implementation of BaseFile should come with its unitest neo/test/unitest/io.
    







"""


# I need to subclass BaseIO
from baseio import BaseIO
# to import : Block, Segment, AnalogSignal, SpikeTrain, SpikeTrainList
from neo.core import *

# So bad :
from numpy import *


# I need to subclass BaseIO
class ExampleIO(BaseIO):
    """
    Class for reading/writing data in a fake file.
    
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
                                ('segmentduration' , { 'value' : 3., 
                                                'label' : 'Segment size (s.)' } ),
                                ('num_segment' , { 'value' : 5,
                                              'label' : 'Segment number' } ),
                                ('num_recordingpoint' , { 'value' : 4,
                                                'label' : 'Number of recording points' } ),
                                ('num_spiketrainbyrecordingpoint' , { 'value' : 3,
                                                'label' : 'Num of spiketrain by recording points' } ),                        
                                ],
                        Segment : [
                                ('segmentduration' , { 'value' : 3., 
                                                'label' : 'Segment size (s.)' } ),
                                ('num_recordingpoint' , { 'value' : 4,
                                                'label' : 'Number of recording points' } ),
                                ('num_spiketrainbyrecordingpoint' , { 'value' : 3,
                                                'label' : 'Num of spiketrain by recording points' } ),
                                    ],
                        }
    
    # do not supported write so no GUI stuf
    write_params       = None
    
    name               = 'example'
    extensions          = [ 'fak' ]
    

    
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
                                        
                                        segmentduration = 3.,
                                        
                                        num_recordingpoint = 4,
                                        num_spiketrainbyrecordingpoint = 2,                        
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
        
        blck = Block()
        
        for i in range(num_segment) :
            # read a segment in the fake file
            # id_segment is just a example it is not taken in account
            seg = self.read_segment(id_segment = i,
                                        segmentduration = segmentduration,
                                        num_recordingpoint = num_recordingpoint,
                                        num_spiketrainbyrecordingpoint = num_spiketrainbyrecordingpoint,
                                        )
            
            # Add seg to blck instance
            blck._segments.append( seg )
        
        return blck
        
    
    # Segment reading is supported so I define this :
    def read_segment(self, 
                                        filename = '',
                                        
                                        num_segment = 12,
                                        id_segment = 0,
                                        name_segment = 'test',
                                        
                                        segmentduration = 3.,
                                        
                                        num_recordingpoint = 4,
                                        num_spiketrainbyrecordingpoint = 2,
                                        
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
        
        """
        
        freq = 10000. #Hz
        t_start = -1.
        num_spike_by_spiketrain = 30
        
        #time vector for generated signal
        t = arange(t_start, t_start+ segmentduration , 1./freq)
        
        # create an empty segment
        seg = Segment()
        
        # create some RecordingPoint :
        for i in range(num_recordingpoint):
            record = RecordingPoint()
            record.name = 'point %i' % i
            
            # Add record to seg instance
            seg._recordingpoints.append( record )
        
        # create some SpikeTrain :
        for i in range(num_recordingpoint):
            for j in range(num_spiketrainbyrecordingpoint):

                spiketr = SpikeTrain()
                
                # There are 2 possibles behaviour for a SpikeTrain
                # holding many Spike instance or directly holding spike times
                # we choose here the second : 
                spiketr._spikes = None
                
                # So we fill the _spike_times attr :
                # generated a random distributed time spike
                spiketr._spike_times = random.rand(num_spike_by_spiketrain)*segmentduration+t_start
                spiketr._spike_times.sort()
                spiketr.t_start = t_start
                spiketr.t_stop = t_start + segmentduration
                
                # for simplification spiketrain is not linked to a neuron instance but it could be
                spiketr.neuron = None
                
                # link this SpikeTrain to its RecordingPoint
                spiketr.recordingpoint = seg._recordingpoints[i]
                
                # this ollowing field is optional and specific from my IO :
                spiketr.ID = 'SpikeTrain %d %d' % (i,j)
                
                # Add spiketr to seg instance
                seg._spiketrains.append( spiketr )
        
        # create some AnalogSignal :
        for i in range(num_recordingpoint):
            anasig = AnalogSignal()
            anasig.freq = freq
            anasig.t_start = t_start
            # choose random freq between 20 and 100 for my sinus signal :
            f1 = random.rand()*80+20.
            # choose a random freq for modulation between .5 and 2
            f2 = random.rand()*1.5+.5
            anasig.signal = sin(2*pi*t*f1) * sin(pi*t*f2)**2
            
            # add very simple spike waveform to the signal
            for j in range(num_spiketrainbyrecordingpoint):
                wsize = int(freq*0.001)*2
                wave = bartlett(wsize/2)*((j+1)*0.5)#+random.rand(wsize)*0.005
                wave = concatenate( (wave,-wave))
                spiketr = seg.get_spiketrains()[i*num_spiketrainbyrecordingpoint+j]
                for ts in spiketr :
                    pos = digitize( [ts] , t )
                    pos = pos[0]-wsize/2
                    if pos>=anasig.signal.size-wsize :
                        pos = anasig.signal.size-wsize-1
                    if pos<0 :
                        pos =0
                    anasig.signal[pos:pos+wsize] +=  wave
            
            
            # link this AnalogSignal to its RecordingPoint
            anasig.recordingpoint = seg._recordingpoints[i]
            
            # theses 2 following fields are optionals and specifics from my IO :
            anasig.unit = 'mV'
            anasig.label = 'fantastic signal %i' % i
            
            # Add anasig to seg instance
            seg._analogsignals.append( anasig )
        
        
        return seg


