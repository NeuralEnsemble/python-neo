# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import rawio
from neo.core import *
from numpy import *
#import pylab

class RawIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        raw = rawio.RawIO()
        seg = raw.read_segment( filename = 'datafiles/File_RAW_1_10kHz_2channels.raw',
                                        samplerate = 10000.,
                                        nbchannel = 2,
                                        bytesoffset = 0,
                                        t_start = 0.,
                                        dtype = 'i2',
                                        rangemin = -10,
                                        rangemax = 10,)
        assert len(seg.get_analogsignals()) ==2
        for sig in seg.get_analogsignals():
            #~ print sig.signal.shape
            assert sig.signal.shape[0] == 200000
            #pylab.plot(sig.signal)
        #pylab.show()
        
    def testWriteReadSinus(self):
        
        seg = Segment()
        freq = 10000.
        t = arange(0,15.,1./freq)
        sig = 3.6*sin(2*numpy.pi*t*60.)
        ana = AnalogSignal( signal = sig,
                                        freq = freq,
                                        )
        seg._analogsignals = [ ana ]
        
        
        raw = rawio.RawIO()
        raw.write_segment(  seg,
                                        filename = 'testNeoRawIO.raw',
                                        dtype = 'f4',
                                        rangemin = -5,
                                        rangemax = 5,
                                        bytesoffset = 0)
        raw2 = rawio.RawIO()
        seg2 = raw2.read_segment(
                                        filename = 'testNeoRawIO.raw',
                                        samplerate = freq,
                                        nbchannel = 1,
                                        bytesoffset = 0,
                                        t_start = 0.,
                                        dtype = 'f4',
                                        rangemin = -5,
                                        rangemax = 5,
                                    )
        ana2 = seg2.get_analogsignals()[0]
        assert len(seg2.get_analogsignals()) == 1
        
        # erreur due dtype
        assert mean((ana2.signal - ana.signal)**2)/mean(ana.signal**2) < .0001
        
        # not possible
        #assert all(ana2.signal == ana.signal)
        
        






if __name__ == "__main__":
    unittest.main()