# -*- coding: utf-8 -*-
import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))
from neo.io import PyNNBinaryIO
from neo.core import *
from numpy import *


class PyNNBinaryIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        io  = PyNNBinaryIO(filename = 'datafiles/File_pynnbinary_1.pynn',)
        seg = io.read()
        assert len(seg.get_spiketrains()) == 4000        
    
    def testOpenFile2(self):
        io  = PyNNBinaryIO(filename = 'datafiles/File_pynnbinary_1.pynn',)
        seg = io.read(id_list=2000)
        assert len(seg.get_spiketrains()) == 2000    
    
    def testOpenFile3(self):
        io  = PyNNBinaryIO(filename = 'datafiles/File_pynnbinary_1.pynn',)
        seg = io.read(id_list=[0,1,2,3], t_start=0, t_stop=1.)
        assert len(seg.get_spiketrains()) == 4
    
    def testNotEmpy(self):
        io  = PyNNBinaryIO(filename = 'datafiles/File_pynnbinary_1.pynn',)
        seg = io.read_spiketrainlist(t_start=0, t_stop=0.5)
        data = seg.mean_rate()
        assert data > 0
    
    def testOpenFile4(self):
        io  = PyNNBinaryIO(filename = 'datafiles/File_pynnbinary_2.pynn',)
        seg = io.read()
        assert len(seg.get_analogsignals()) == 2
        for sig in seg.get_analogsignals():
            assert sig.signal.shape[0] == 10001            
    
    #def testWriteReadSinusFloat(self):
        
        #seg = Segment()
        #freq = 10000.
        #t = arange(0,15.,1./freq)
        #sig = 3.6*sin(2*numpy.pi*t*60.)
        #ana = AnalogSignal( signal = sig,
                                        #freq = freq,
                                        #)
        #seg._analogsignals = [ ana , ana]
        
        
        #io = RawIO(filename = 'testNeoRawIO.raw',)
        #io.write_segment(  seg,
                                        #dtype = 'f4',
                                        #rangemin = -5,
                                        #rangemax = 5,
                                        #bytesoffset = 0)
        #io = RawIO(filename = 'testNeoRawIO.raw',)
        #seg2 = io.read_segment(
                                        #samplerate = freq,
                                        #nbchannel = 2,
                                        #bytesoffset = 0,
                                        #t_start = 0.,
                                        #dtype = 'f4',
                                        #rangemin = -5,
                                        #rangemax = 5,
                                    #)
        #ana2 = seg2.get_analogsignals()[0]
        #assert len(seg2.get_analogsignals()) == 2
        
        ## erreur due dtype
        #assert mean((ana2.signal - ana.signal)**2)/mean(ana.signal**2) < .00001
        
        ## not possible
        ##assert all(ana2.signal == ana.signal)
        
        
    #def testWriteReadSinusint(self):
        
        #seg = Segment()
        #freq = 10000.
        #t = arange(0,15.,1./freq)
        #sig = 3.6*sin(2*numpy.pi*t*60.)
        #ana = AnalogSignal( signal = sig,
                                        #freq = freq,
                                        #)
        #seg._analogsignals = [ ana , ana]
        
        
        #io = RawIO(filename = 'testNeoRawIO.raw',)
        #io.write_segment(  seg,
                                        #dtype = 'i2',
                                        #rangemin = -5,
                                        #rangemax = 5,
                                        #bytesoffset = 0)
        #io = RawIO(filename = 'testNeoRawIO.raw',)
        #seg2 = io.read_segment(
                                        #samplerate = freq,
                                        #nbchannel = 2,
                                        #bytesoffset = 0,
                                        #t_start = 0.,
                                        #dtype = 'i2',
                                        #rangemin = -5,
                                        #rangemax = 5,
                                    #)
        #ana2 = seg2.get_analogsignals()[0]
        #assert len(seg2.get_analogsignals()) == 2
        
        ## erreur due dtype
        #assert mean((ana2.signal - ana.signal)**2)/mean(ana.signal**2) < .0001
        
        ## not possible
        ##assert all(ana2.signal == ana.signal)





if __name__ == "__main__":
    unittest.main()