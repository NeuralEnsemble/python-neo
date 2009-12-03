# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import elanio
from neo.core import *
from numpy import *
#import pylab

class ElanIOTest(unittest.TestCase):
    
#    def testOpenFile1(self):
#        elan = elanio.ElanIO()
#        seg = elan.read_segment( filename = 'datafiles/File_elan_1.eeg',)
#        assert len(seg.get_analogsignals()) ==6
#        for sig in seg.get_analogsignals():
#            assert sig.signal.shape[0] == 1082785
##            pylab.plot(sig.t(),sig.signal)
#        
#        assert len (seg.get_events() )==47 
##        for ev in seg.get_events():
##            print ev.time
##            pylab.axvline(ev.time)
#        
#        #pylab.show()
        
    def testWriteReadSinus(self):
        
        seg = Segment()
        freq = 10000.
        t = arange(0,15.,1./freq)
        sig = 3.6*sin(2*numpy.pi*t*60.)
        ana = AnalogSignal( signal = sig,
                                        freq = freq,
                                        )
        seg._analogsignals = [ ana ]
        
        
        elan = elanio.ElanIO()
        elan.write_segment(  seg,
                            filename = 'testNeoElanIO.raw',
                            )
        elan2 = elanio.ElanIO()
        seg2 = elan2.read_segment(
                            filename = 'testNeoElanIO.raw',
                            )
        ana2 = seg.get_analogsignals()[0]
        assert len(seg2.get_analogsignals()) == 1
        assert all(ana2.signal == ana.signal)
        






if __name__ == "__main__":
    unittest.main()