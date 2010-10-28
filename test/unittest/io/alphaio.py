# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import AlphaIO
from neo.core import *
from numpy import *
from scipy import rand
import pylab
import time

class AlphaIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        r = AlphaIO(filename =   'datafiles/File_AlphaOmega_1.map')

        t1 = time.time()
        blck = r.read_block()
        t2 = time.time()
        print 'import time', t2-t1
        
        for seg in blck.get_segments() :
            print ''
            fig = pylab.figure()
            ax = fig.add_subplot(1,1,1)
            
            #print len(seg.get_analogsignals())
            #~ assert len(seg.get_analogsignals()) ==2
            
            for s,sig in enumerate(seg.get_analogsignals()):
                #print sig.signal.shape[0], sig.sampling_rate
                #assert sig.name == 'ImRK01G20'
                #assert sig.unit == 'pA'
                #~ assert sig.signal.shape[0] == 49920
                print sig.name, sig.channel, sig.t_start, sig.sampling_rate, sig.signal.size, sig.signal.dtype, sig.t_stop
                if s==4:
                    ax.plot(sig.t()[:] , sig.signal[:] , color = colors[s%7] )
                
                
            
            #print len (seg.get_events() )
            #assert len (seg.get_events() )==0
            dict_color = { }
            for ev in seg.get_events():
                ax.axvline(ev.time, color = 'b')
                
        pylab.show()

if __name__ == "__main__":
    unittest.main()


