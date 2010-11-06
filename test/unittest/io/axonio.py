# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import AxonIO
from neo.core import *
from numpy import *
from scipy import rand
import pylab


class AxonIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        io = AxonIO(filename = 'datafiles/File_axon_1.abf')
        block = io.read_block( )
        for seg in block.get_segments() :
            #print len(seg.get_analogsignals())
            assert len(seg.get_analogsignals()) ==1
            for sig in seg.get_analogsignals():
                #print sig.signal.shape[0], sig.name, sig.num, sig.unit, sig.sampling_rate
                assert sig.name == 'ImRK01G20'
                assert sig.unit == 'pA'
                assert sig.signal.shape[0] == 1912832
            
            #print len (seg.get_events() )
            assert len (seg.get_events() )==0
            for ev in seg.get_events():
                pass

    def testOpenFile2(self):
        io = AxonIO(filename = 'datafiles/File_axon_2.abf')
        block = io.read_block( )
        for seg in block.get_segments() :
            #print len(seg.get_analogsignals())
            assert len(seg.get_analogsignals()) ==1
            for sig in seg.get_analogsignals():
                #print sig.signal.shape[0], sig.name, sig.num, sig.unit, sig.sampling_rate
                #assert sig.name == 'ImRK01G20'
                #assert sig.unit == 'pA'
                assert sig.sampling_rate == 1000.0
                assert sig.signal.shape[0] == 1200000
            
            #print len (seg.get_events() )
            assert len (seg.get_events() )==4
            for ev in seg.get_events():
                pass

    def testOpenFile3(self):
        io = AxonIO(filename = 'datafiles/File_axon_3.abf')
        block = io.read_block( )
        
        for seg in block.get_segments() :
            #print len(seg.get_analogsignals())
            assert len(seg.get_analogsignals()) ==2
            for sig in seg.get_analogsignals():
                #print sig.signal.shape[0], sig.name, sig.num, sig.unit, sig.sampling_rate
                #assert sig.name == 'ImRK01G20'
                #assert sig.unit == 'pA'
                assert sig.sampling_rate == 20000.0
                assert sig.signal.shape[0] == 20644
            
            #print len (seg.get_events() )
            assert len (seg.get_events() )==0
            for ev in seg.get_events():
                pass


    def testOpenFile4(self):

        io = AxonIO(filename = 'datafiles/File_axon_4.abf')
        block = io.read_block( )

        for seg in block.get_segments() :
            #print len(seg.get_analogsignals())
            assert len(seg.get_analogsignals()) ==1
            for sig in seg.get_analogsignals():
                #print sig.signal.shape[0], sig.name, sig.num, sig.unit, sig.sampling_rate
                assert sig.name == 'ImRK01G20'
                assert sig.unit == 'pA'
                assert sig.sampling_rate == 10000.0
                assert sig.signal.shape[0] == 2176512
            
            #print len (seg.get_events() )
            assert len (seg.get_events() )==1
            for ev in seg.get_events():
                #print ev.time
                #print ev.name
                assert ev.name == 'drogue on'





if __name__ == "__main__":
    unittest.main()
    

