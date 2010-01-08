# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import spike2io
from neo.core import *
from numpy import *
from scipy import rand
import pylab

class Spike2IOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        spike2 = spike2io.Spike2IO()
        block = spike2.read_block( filename = 'datafiles/File_spike2_1.smr',)
        print len(block.get_segments())
        
        for seg in block.get_segments() :
            print len(seg.get_analogsignals())
            #assert len(seg.get_analogsignals()) ==1
            for sig in seg.get_analogsignals():
                print sig.signal.shape[0], sig.freq
                #assert sig.name == 'ImRK01G20'
                #assert sig.unit == 'pA'
                #assert sig.signal.shape[0] == 1912832
            
            print len (seg.get_events() )
            #assert len (seg.get_events() )==0
            for ev in seg.get_events():
                pass 
        #pylab.show()


if __name__ == "__main__":
    unittest.main()