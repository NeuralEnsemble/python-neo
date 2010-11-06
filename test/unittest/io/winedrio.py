# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import WinEdrIO
from neo.core import *
from numpy import *
from scipy import rand
import pylab

colors = [  'b' , 'r' ,'g' , 'y' , 'k' , 'm' , 'c']

class WinEdrIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        
        io = WinEdrIO(filename = 'datafiles/File_WinEDR_1.EDR',)
        seg = io.read_segment()
        
        fig = pylab.figure()
        ax = fig.add_subplot(1,1,1)
            
        #print len(seg.get_analogsignals())
        #~ assert len(seg.get_analogsignals()) ==2
        
        for s,sig in enumerate(seg.get_analogsignals()):
            #print sig.signal.shape[0], sig.sampling_rate
            #assert sig.name == 'ImRK01G20'
            #assert sig.unit == 'pA'
            #~ assert sig.signal.shape[0] == 49920
            ax.plot(sig.t() , sig.signal , color = colors[s%7] )
            
        
                
        pylab.show()


if __name__ == "__main__":
    unittest.main()


