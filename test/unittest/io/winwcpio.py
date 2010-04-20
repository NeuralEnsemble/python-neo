# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import WinWcpIO
from neo.core import *
from numpy import *
from scipy import rand
import pylab

colors = [  'b' , 'r' ,'g' , 'y' , 'k' , 'm' , 'c']

class WinWcpIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        
        io = WinWcpIO(filename = 'datafiles/File_winwcp_1.wcp',)
        blck = io.read_block()
        
        fig = pylab.figure()
        ax = fig.add_subplot(1,1,1)
        for seg in blck.get_segments() :
            
            #print len(seg.get_analogsignals())
            assert len(seg.get_analogsignals()) ==2
            
            for s,sig in enumerate(seg.get_analogsignals()):
                #print sig.signal.shape[0], sig.sampling_rate
                #assert sig.name == 'ImRK01G20'
                #assert sig.unit == 'pA'
                assert sig.signal.shape[0] == 49920
                ax.plot(sig.t() , sig.signal , color = colors[s%7] )
                
            
            #print len (seg.get_events() )
            #assert len (seg.get_events() )==0
            dict_color = { }
            for ev in seg.get_events():
                ax.axvline(ev.time, color = 'b')
                
        pylab.show()


if __name__ == "__main__":
    unittest.main()


