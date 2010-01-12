# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import asciispikeio
from neo.core import *
from numpy import *
from scipy import rand
import pylab

class AsciiSpikeIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        
        asciispike = asciispikeio.AsciiSpikeIO()
        seg = asciispike.read_segment( filename = 'datafiles/File_ascii_spiketrain_1.txt',
                                        delimiter = '\t',
                                        t_start = 0.,
                                    )
        
        fig = pylab.figure()
        ax = fig.add_subplot(1,1,1)
        for s,spiketr in enumerate(seg.get_spiketrains()) :
            ts = spiketr.spike_times
            ax.plot( ts , ones_like(ts)*s ,
                        linestyle = '',
                        marker = '|' ,
                        markersize = 5)
            
                
        #pylab.show()

    def testWriteReadSpikeTrain(self):

        seg = Segment()
        
        spiketr1 = SpikeTrain(spike_times = rand(50)*3., t_start =0)
        spiketr2 = SpikeTrain(spike_times = rand(50)*3., t_start =0)
        seg._spiketrains = [ spiketr1 , spiketr2 ]
        
        
        asciispike = asciispikeio.AsciiSpikeIO()
        asciispike.write_segment( seg,
                                filename = 'testNeoAsciiSpikelIO.txt',
                                delimiter = '\t',)
        
        asciispike2 = asciispikeio.AsciiSpikeIO()
        seg2 = asciispike2.read_segment(
                                    filename = 'testNeoAsciiSpikelIO.txt',
                                    delimiter = '\t',)

        
        assert len(seg2.get_spiketrains() ) == 2
        
        assert all( abs(seg2.get_spiketrains()[0].spike_times - spiketr1.spike_times) < 0.000001 )
        assert all( abs(seg2.get_spiketrains()[1].spike_times - spiketr2.spike_times) < 0.000001 )
        



if __name__ == "__main__":
    unittest.main()
