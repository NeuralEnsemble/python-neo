# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import tryitio
from neo.core import *
from numpy import *
from scipy import rand
import pylab

class TryItIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        
        tryit = tryitio.TryItIO()
        block = tryit.read_block( num_segment = 2,
                                    
                                    segmentduration = 3.,
                                    
                                    num_recordingpoint = 4,
                                    num_spiketrainbyrecordingpoint = 2,
                                    )
        
        for seg in block.get_segments() :
            fig = pylab.figure()
            ax = fig.add_subplot(2,1,1)
            #print len(seg.get_analogsignals())
            #assert len(seg.get_analogsignals()) ==64
            for sig in seg.get_analogsignals():
                #print sig.signal.shape[0]
                assert sig.signal.shape[0] == 30000
                ax.plot(sig.t(),sig.signal)
                #print sig.num, sig.label , sig.ground
            
            ax = fig.add_subplot(2,1,2 , sharex = ax)
            for s,spiketr in enumerate(seg.get_spiketrains()) :
                ts = spiketr.spike_times
                ax.plot( ts , ones_like(ts)*s ,
                            linestyle = '',
                            marker = '|' ,
                            markersize = 5)
            
            #print len (seg.get_events() )
            #assert len (seg.get_events() )==47 
            for ev in seg.get_events():
                #print ev.time
                ax.axvline(ev.time)
                
        pylab.show()



if __name__ == "__main__":
    unittest.main()
