# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import spike2io
from neo.core import *
from numpy import *
from scipy import rand
import pylab

class Spike2IOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        spike2 = spike2io.Spike2IO()
        block = spike2.read_block( filename = 'datafiles/File_spike2_1.smr',)
#        block = spike2.read_block( filename = 'datafiles/R05-C05C.SMR',)
#        block = spike2.read_block( filename = 'datafiles/R12-C10b.smr',)
#        block = spike2.read_block( filename = 'datafiles/R12-C11f.smr',)
#        block = spike2.read_block( filename = 'datafiles/R14-C02C.SMR',)
#        block = spike2.read_block( filename = 'datafiles/R21-C11d.smr',)
#        block = spike2.read_block( filename = 'datafiles/R25-C07C.SMR',)
#        block = spike2.read_block( filename = 'datafiles/J0_G3S2.SMR',)
        #block = spike2.read_block( filename = 'datafiles/example.smr',)
        #block = spike2.read_block( filename = 'datafiles/data spike2 4channels 1 hypno 1textmarkchannel 1markerchannel.smr',)


        
        print len(block.get_segments())
        
        for seg in block.get_segments() :
            fig = pylab.figure()
            ax = fig.add_subplot(2,1,1)
            #print len(seg.get_analogsignals())
            #assert len(seg.get_analogsignals()) ==1
            for sig in seg.get_analogsignals():
                #print sig.signal.shape[0], sig.freq
                #assert sig.name == 'ImRK01G20'
                #assert sig.unit == 'pA'
                #assert sig.signal.shape[0] == 1912832
                ax.plot(sig.t() , sig.signal)
                
            
            #print len (seg.get_events() )
            #assert len (seg.get_events() )==0
            for ev in seg.get_events():
                ax.axvline(ev.time)
                pass 
            
            ax = fig.add_subplot(2,1,2)
            for s,spiketr in enumerate(seg.get_spiketrains()) :
                ts = spiketr.spike_times
                ax.plot( ts , ones_like(ts)*s ,
                            linestyle = '',
                            marker = '|' ,
                            markersize = 5)
                fig2 = pylab.figure()
                ax2 = fig2.add_subplot(2,1,1)
                for spike in spiketr.get_spikes():
                    tv = arange(spike.waveform.size , dtype = 'f')/spike.waveform.size*spiketr.freq+spike.time
                    ax2.plot(tv,spike.waveform)

                
        pylab.show()


if __name__ == "__main__":
    unittest.main()


