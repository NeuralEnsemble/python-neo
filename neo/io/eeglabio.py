# -*- coding: utf-8 -*-
"""

Classe for reading/writing data from EEGLAB.
EEGLAB is a software written in Matlab for analysing EEG datasets.
The format is based on Matlab files.

Supported : Read/Write

@author : sgarcia

"""


from baseio import BaseIO
#from neo.core import *
from ..core import *
from numpy import *
import numpy
import datetime

from scipy import io


class EegLabIO(BaseIO):
    """
    Classe for reading/writing data from EEGLab.
    base on matlab format
    
    **Usage**

    **Example**
        #read a file
        io = EegLabIO(filename = 'myfile.set')
        seg = io.read() # read the entire file
        
        seg is a Segment that contains AnalogSignals and Events
        
        # write a file
        io = EegLabIO(filename = 'myfile.set')
        seg = Segment()
        io.write(seg)    
    
    """
    
    is_readable        = True
    is_writable        = True

    supported_objects  = [ Segment , AnalogSignal , Event]
    readable_objects   = [Segment]
    writeable_objects  = [Segment]  

    has_header         = False
    is_streameable     = False
    read_params        = { Segment : [ ] }
    write_params       = { Segment : [ ] }
    
    name               = None
    extensions         = [ 'set' ]
    
    def __init__(self , filename = None) :
        """
        This class read/write a eeglab matlab based file.
        
        **Arguments**
            filename : the filename to read
        """
        BaseIO.__init__(self)
        self.filename = filename

    def read(self , **kargs):
        """
        Read the file.
        Return a neo.Segment
        See read_segment for detail.
        """
        return self.read_segment( **kargs)
    
    def read_segment(self, ):
        """
        read a segment in a .set file.
        segment contains AnalogSignal and Event
        
        **Arguments**
            no arguments
        """
        
        seg = Segment()
        
        d = io.loadmat(self.filename,appendmat=False , struct_as_record = True , squeeze_me = False)
        EEG = d['EEG']
        EEG = EEG[0,0]
        
#        print type(EEG)
#        print EEG.dtype
#        print EEG.shape
#        print type(EEG['chanlocs'])
#        print EEG['chanlocs'].shape
#        print EEG['chanlocs'].dtype
        
        chanlocs = EEG['chanlocs'].squeeze()
        data = EEG['data']
        srate = EEG['srate'][0,0]
        events = EEG['event'].squeeze()
        
        
        for e in xrange( data.shape[0] ) :
            anaSig = AnalogSignal()
            anaSig.label = chanlocs['labels'][e]
            anaSig.freq = srate
            anaSig.signal = data[e,:]
            anaSig.unit = 'microV'
            seg._analogsignals.append( anaSig )
            
        for e in xrange(events.size) :
            event = Event()
            event.time = float(events['latency'][e])/srate
            event.label = events['latency'][e]
            event.num = e
            seg._events.append( event )
        
        return seg
        
        
    def write(self , *args , **kargs):
        """
        Write segment in a file.
        See write_segment for detail.
        """
        self.write_segment(*args , **kargs)

    def write_segment(self, segment,):
        """
        
         **Arguments**
            segment : the segment to write. Only analog signals and events will be written.
        """
        seg = segment
        
        #data
        data = zeros( (len(seg.get_analogsignals()) , seg.get_analogsignals()[0].signal.size) , 'f4')
        for i,anaSig in enumerate(seg.get_analogsignals()):
            data[i,:] = anaSig.signal
        
        #srate
        srate = array( [[seg.get_analogsignals()[0].freq ]])
        
        #event
        event = zeros( (1, len(seg.get_events())) , 
                                       dtype = [('type', '|O8'),
                                                ('latency', '|O8'),
                                                ('urevent', '|O8')]
                                            )
        for i, ev in enumerate(seg.get_events()):
            event[0,i]['latency'] = int(ev.time*srate)
            event[0,i]['type'] = ev.label
        
        #canlocs
        chanlocs = zeros ( (1, len(seg.get_analogsignals())) , 
                                dtype = [('theta', '|O8'),
                                         ('radius', '|O8'),
                                         ('labels', '|O8'),
                                         ('sph_theta', '|O8'),
                                         ('sph_phi', '|O8'),
                                         ('X', '|O8'),
                                         ('Y', '|O8'),
                                         ('Z', '|O8'),
                                         ('urchan', '|O8')]
                            )
        for i,anaSig in enumerate(seg.get_analogsignals()):
            chanlocs[0,i]['labels'] = anaSig.label
        
        EEG = array( [[ (data ,
                        srate,
                        event,
                        chanlocs,
                        )]] ,
                    dtype =[    ('data' , 'O') ,
                                ('srate' , 'O') ,
                                ('event' , 'O') ,
                                ('chanlocs' , 'O') ,
                                    ])
        
        io.savemat(self.filename , { 'EEG' : EEG }  , appendmat = False)
        
