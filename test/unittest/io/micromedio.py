# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import MicromedIO
from neo.core import *
from numpy import *
from scipy import rand
import pylab

class MicromedIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        io = MicromedIO( filename = 'datafiles/File_micromed_1.TRC',)
        seg = io.read_segment()
        #print len(seg.get_analogsignals())
        assert len(seg.get_analogsignals()) ==64
        for sig in seg.get_analogsignals():
            #print sig.signal.shape[0]
            #assert sig.signal.shape[0] == 410160
            pylab.plot(sig.t(),sig.signal)
            #print sig.num, sig.label , sig.ground
            pass
        
        print len (seg.get_events() )
        #assert len (seg.get_events() )==47 
        for ev in seg.get_events():
            #print ev.time
            pylab.axvline(ev.time)
        
        pylab.show()
    

if __name__ == "__main__":
    unittest.main()
