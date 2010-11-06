# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import RawIO
from neo.core import *
from numpy import *
#import pylab

class RawIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        io = RawIO(filename = 'datafiles/File_RAW_1_10kHz_2channels.raw',)
        seg = io.read_segment( 
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
        
    def testWriteReadSinusFloat(self):
        
        seg = Segment()
        sampling_rate = 10000.
        t = arange(0,15.,1./sampling_rate)
        sig = 3.6*sin(2*numpy.pi*t*60.)
        ana = AnalogSignal( signal = sig,
                                        sampling_rate = sampling_rate,
                                        )
        seg._analogsignals = [ ana , ana]
        
        
        io = RawIO(filename = 'testNeoRawIO.raw',)
        io.write_segment(  seg,
                                        dtype = 'f4',
                                        rangemin = -5,
                                        rangemax = 5,
                                        bytesoffset = 0)
        io = RawIO(filename = 'testNeoRawIO.raw',)
        seg2 = io.read_segment(
                                        samplerate = sampling_rate,
                                        nbchannel = 2,
                                        bytesoffset = 0,
                                        t_start = 0.,
                                        dtype = 'f4',
                                        rangemin = -5,
                                        rangemax = 5,
                                    )
        ana2 = seg2.get_analogsignals()[0]
        assert len(seg2.get_analogsignals()) == 2
        
        # erreur due dtype
        assert mean((ana2.signal - ana.signal)**2)/mean(ana.signal**2) < .00001
        
        # not possible
        #assert all(ana2.signal == ana.signal)
        
        
    def testWriteReadSinusint(self):
        
        seg = Segment()
        sampling_rate = 10000.
        t = arange(0,15.,1./sampling_rate)
        sig = 3.6*sin(2*numpy.pi*t*60.)
        ana = AnalogSignal( signal = sig,
                                        sampling_rate = sampling_rate,
                                        )
        seg._analogsignals = [ ana , ana]
        
        
        io = RawIO(filename = 'testNeoRawIO.raw',)
        io.write_segment(  seg,
                                        dtype = 'i2',
                                        rangemin = -5,
                                        rangemax = 5,
                                        bytesoffset = 0)
        io = RawIO(filename = 'testNeoRawIO.raw',)
        seg2 = io.read_segment(
                                        samplerate = sampling_rate,
                                        nbchannel = 2,
                                        bytesoffset = 0,
                                        t_start = 0.,
                                        dtype = 'i2',
                                        rangemin = -5,
                                        rangemax = 5,
                                    )
        ana2 = seg2.get_analogsignals()[0]
        assert len(seg2.get_analogsignals()) == 2
        
        # erreur due dtype
        assert mean((ana2.signal - ana.signal)**2)/mean(ana.signal**2) < .0001
        
        # not possible
        #assert all(ana2.signal == ana.signal)





if __name__ == "__main__":
    unittest.main()