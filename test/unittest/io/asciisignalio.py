# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import AsciiSignalIO
from neo.core import *
from numpy import *
import pylab



class AsciiSignalIOTest(unittest.TestCase):
    def testOpenFile1(self):
        io = AsciiSignalIO(filename = 'datafiles/File_ascii_1.asc')
        seg = io.read_segment( 
                                        delimiter = '  ',
                                        usecols = None,
                                        skiprows =11,
                                        timecolumn = None,
                                        samplerate = 512.,
                                        t_start = 0.,
                                        method = 'homemade'
                                    )
        assert len(seg.get_analogsignals()) == 8
        for sig in seg.get_analogsignals():
            assert sig.signal.shape[0] == 79360
#            pylab.plot(sig.signal)
#        pylab.show()

    def testOpenFile2(self):
        io = AsciiSignalIO(filename = 'datafiles/File_ascii_2.txt')
        seg = io.read_segment( 
                                        delimiter = '  ',
                                        usecols = None,
                                        skiprows =0,
                                        timecolumn = 0,
                                        method = 'homemade'
                                        )
        assert len(seg.get_analogsignals()) == 30
        for sig in seg.get_analogsignals():
            assert sig.signal.shape[0] == 18749
#            pylab.plot(sig.signal)
#        pylab.show()

    def testOpenFile3(self):
        io = AsciiSignalIO(filename = 'datafiles/File_ascii_3.txt')
        seg = io.read_segment( 
                                        delimiter = '\t',
                                        usecols = None,
                                        skiprows =1,
                                        timecolumn = 0,
                                        method = 'homemade'
                                        )
        assert len(seg.get_analogsignals()) == 7
        for sig in seg.get_analogsignals():
            assert sig.signal.shape[0] == 55800
#            pylab.plot(sig.signal)
#        pylab.show()

    def testWriteReadSinus(self):

        seg = Segment()
        freq = 10000.
        t = arange(0,15.,1./freq)
        sig = 3.6*sin(2*numpy.pi*t*60.)
        ana = AnalogSignal( signal = sig,
                                        freq = freq,
                                        )
        seg._analogsignals = [ ana ]
        
        
        io1 = AsciiSignalIO(filename = 'testNeoAsciiIO.txt',)
        io1.write_segment(  seg,
                                timecolumn = 1,
                                delimiter = '\t',
                                )
        
        io2 = AsciiSignalIO(filename = 'testNeoAsciiIO.txt',)
        seg2 = io2.read_segment(
                                    delimiter = '\t',
                                    timecolumn = 1,
                                    method = 'genfromtxt'
                                    )
        ana2 = seg2.get_analogsignals()[0]
        # 2% erreur due to i2 convertion
        #print mean((ana2.signal - ana.signal)**2)/mean(ana.signal**2)
        assert mean((ana2.signal - ana.signal)**2)/mean(ana.signal**2) < .01
        
        # not possible
        #assert all(ana2.signal == ana.signal)



if __name__ == "__main__":
    unittest.main()