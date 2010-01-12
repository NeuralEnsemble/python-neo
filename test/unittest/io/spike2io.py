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
        seg = spike2.read_segment( filename = 'datafiles/File_spike2_1.smr',transform_event_to_spike = [3])
#        block = spike2.read_segment( filename = 'datafiles/R05-C05C.SMR',)
#        block = spike2.read_segment( filename = 'datafiles/R12-C10b.smr',)
#        block = spike2.read_segment( filename = 'datafiles/R12-C11f.smr',)
#        block = spike2.read_segment( filename = 'datafiles/R14-C02C.SMR',)
#        block = spike2.read_segment( filename = 'datafiles/R21-C11d.smr',)
#        block = spike2.read_segment( filename = 'datafiles/R25-C07C.SMR',)
#        block = spike2.read_segment( filename = 'datafiles/J0_G3S2.SMR',)
        #block = spike2.read_segment( filename = 'datafiles/example.smr',)
        #block = spike2.read_segment( filename = 'datafiles/20091007_000.smr',)
        #block = spike2.read_segment( filename = 'datafiles/20091103_000.smr',)
        #block = spike2.read_segment( filename = 'datafiles/20091104_000.smr',)
        
        #block = spike2.read_segment( filename = 'datafiles/data spike2 4channels 1 hypno 1textmarkchannel 1markerchannel.smr',)


        fig = pylab.figure()
        ax = fig.add_subplot(4,1,1)
        ax2 = fig.add_subplot(4,1,2, sharex =ax)
        ax3 = fig.add_subplot(4,1,3, sharex =ax)
        ax4 = fig.add_subplot(4,1,4, sharex =ax)
        #print len(seg.get_analogsignals())
        #assert len(seg.get_analogsignals()) ==1
        for sig in seg.get_analogsignals():
            #print sig.signal.shape[0], sig.freq
            #assert sig.name == 'ImRK01G20'
            #assert sig.unit == 'pA'
            #assert sig.signal.shape[0] == 1912832
            
            
            ax.plot(sig.t()[:1e5] , sig.signal[:1e5])
            
        
        #print len (seg.get_events() )
        #assert len (seg.get_events() )==0
        for ev in seg.get_events():
            if hasattr(ev , 'waveform'):
                tv = arange(ev.waveform.size , dtype = 'f')/ev.freq+ev.time
                ax2.plot(tv,ev.waveform)
                ax2.axvline(ev.time)
            else :
                ax.axvline(ev.time)
            
            
        
        for s,spiketr in enumerate(seg.get_spiketrains()) :
            print 'spiketrain' , s
            ts = spiketr.spike_times
            print ts
            ax3.plot( ts , ones_like(ts)*s ,
                        linestyle = '',
                        marker = '|' ,
                        markersize = 5)
            
            for spike in spiketr.get_spikes():
                if spike.waveform is not None:
                    tv = arange(spike.waveform.size , dtype = 'f')/spiketr.freq+spike.time
                    ax4.plot(tv,spike.waveform)

                
        pylab.show()


if __name__ == "__main__":
    unittest.main()


