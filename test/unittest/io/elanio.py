# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import ElanIO
from neo.core import *
from numpy import *
from scipy import rand
#import pylab

class ElanIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        io = ElanIO(filename = 'datafiles/File_elan_1.eeg',)
        seg = io.read_segment( )
        assert len(seg.get_analogsignals()) ==4
        for sig in seg.get_analogsignals():
            assert sig.signal.shape[0] == 1082785
#            pylab.plot(sig.t(),sig.signal)
        
        assert len (seg.get_events() )==47 
#        for ev in seg.get_events():
#            print ev.time
#            pylab.axvline(ev.time)
        
        #pylab.show()
    
    def testWriteReadSinusAndEvent(self):
        
        seg = Segment()
        sampling_rate = 10000.
        totaltime = 15.
        t = arange(0,totaltime,1./sampling_rate)
        sig = 3.6*sin(2*numpy.pi*t*60.)
        ana = AnalogSignal( signal = sig,
                                        sampling_rate = sampling_rate,
                                        )
        seg._analogsignals = [ ana , ana ]
        nbevent = 40
        for i in range(nbevent):
            seg._events += [ Event(time = rand()*totaltime) ]
            
        
        io = ElanIO(filename = 'testNeoElanIO.eeg',)
        io.write_segment(  seg,)
        io = ElanIO(filename = 'testNeoElanIO.eeg',)
        seg2 = io.read_segment()
        ana2 = seg2.get_analogsignals()[0]

        assert len(seg2.get_analogsignals()) == 2
        
        # 2% erreur due to i2 convertion
        #print mean((ana2.signal - ana.signal)**2)/mean(ana.signal**2) 
        assert mean((ana2.signal - ana.signal)**2)/mean(ana.signal**2) < .02
        
        for i in range(nbevent):
            assert seg._events[i].time - seg2._events[i].time < 1./sampling_rate






if __name__ == "__main__":
    unittest.main()