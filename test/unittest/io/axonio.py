# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import axonio
from neo.core import *
from numpy import *
from scipy import rand
import pylab

axon = axonio.AxonIO()
block = axon.read_block( filename = 'datafiles/File_axon_1.abf',)

class AxonIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        axon = axonio.AxonIO()
        block = axon.read_block( filename = 'datafiles/File_axon_1.abf',)
        seg = block.get_segments()[0]
        print len(seg.get_analogsignals())
        #assert len(seg.get_analogsignals()) ==7
        for sig in seg.get_analogsignals():
            print sig.signal.shape[0]
            #assert sig.signal.shape[0] == 240000
            pylab.plot(sig.t(),sig.signal)
            #print sig.num, sig.label , sig.ground
        
        print len (seg.get_events() )
        #assert len (seg.get_events() )==47 
        for ev in seg.get_events():
            #print ev.time
            pylab.axvline(ev.time)
        
        pylab.show()

if __name__ == "__main__":
    #unittest.main()
    pass

