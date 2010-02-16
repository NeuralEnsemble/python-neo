# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import elphyio
from neo.core import *
from numpy import *
from scipy import rand
import pylab



class ElphyIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        elphy = elphyio.ElphyIO()
        block = elphy.read_block( filename = 'datafiles/File_elphy_1.DAT',)
        seg = block.get_segments()[0]
        #print len(seg.get_analogsignals())
        assert len(seg.get_analogsignals()) ==7
        for sig in seg.get_analogsignals():
            #print sig.signal.shape[0]
            assert sig.signal.shape[0] == 240000
            pylab.plot(sig.t(),sig.signal)
            #print sig.num, sig.label , sig.ground
        
        print len (seg.get_events() )
        #assert len (seg.get_events() )==47 
        for ev in seg.get_events():
            #print ev.time
            pylab.axvline(ev.time)
        
        pylab.show()
    
#    def testWriteReadSinusAndEvent(self):
#        
#        seg = Segment()
#        freq = 10000.
#        totaltime = 15.
#        t = arange(0,totaltime,1./freq)
#        sig = 3.6*sin(2*numpy.pi*t*60.)
#        ana = AnalogSignal( signal = sig,
#                                        freq = freq,
#                                        )
#        nbchannel = 16
#        for i in range(nbchannel) :
#            seg._analogsignals += [ ana ]
#        nbevent = 40
#        for i in range(nbevent):
#            seg._events += [ Event(time = rand()*totaltime) ]
#            
#        
#        elphy = elphyio.ElphyIO()
#        elphy.write_segment(  seg,
#                            filename = 'testNeoElphyIO.TRC',
#                            )
#        
#        elphy2 = elphyio.ElphyIO()
#        seg2 = elphy2.read_segment(
#                            filename = 'testNeoElphyIO.TRC',
#                            )
#        ana2 = seg2.get_analogsignals()[0]
#        
#        assert len(seg2.get_analogsignals()) == nbchannel
#        
#        # 1% erreur due to i2 convertion
#        print mean((ana2.signal - ana.signal)**2)/mean(ana.signal**2)
#        assert mean((ana2.signal - ana.signal)**2)/mean(ana.signal**2) < .01
#        
#        for i in range(nbevent):
#            assert seg._events[i].time - seg2._events[i].time < 1./freq
#

if __name__ == "__main__":
    unittest.main()

