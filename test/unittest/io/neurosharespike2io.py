# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import NeuroshareSpike2IO
from neo.core import *
from numpy import *
from scipy import rand
import pylab

class NeuroshareSpike2IOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        spike2 = NeuroshareSpike2IO(filename = 'datafiles/File_spike2_1.smr')
        seg = io.read_segment( )
        
        fig = pylab.figure()
        ax = fig.add_subplot(4,1,1)
        ax2 = fig.add_subplot(4,1,2, sharex =ax)
        ax3 = fig.add_subplot(4,1,3, sharex =ax)
        ax4 = fig.add_subplot(4,1,4, sharex =ax)
        print len(seg.get_analogsignals())
        #assert len(seg.get_analogsignals()) ==1
        colors = [  'b' , 'r' ,'g' , 'y' , 'k' , 'm' , 'c']
        for s,sig in enumerate(seg.get_analogsignals()):
            #print sig.signal.shape[0], sig.freq
            #assert sig.name == 'ImRK01G20'
            #assert sig.unit == 'pA'
            #assert sig.signal.shape[0] == 1912832
            ax.plot(sig.t()[:1e6] , sig.signal[:1e6] , color = colors[s%7] )
            
        
        #print len (seg.get_events() )
        #assert len (seg.get_events() )==0
        dict_color = { }
        for ev in seg.get_events():
            if ev.type not in dict_color.keys():
                dict_color[ev.type] = colors[len(dict_color) % 7]
            if hasattr(ev , 'waveform'):
                tv = arange(ev.waveform.size , dtype = 'f')/ev.freq+ev.time
                ax2.plot(tv,ev.waveform[:,0] , color = dict_color[ev.type])
                ax2.axvline(ev.time , color = dict_color[ev.type])
            else :
                ax.axvline(ev.time, color = dict_color[ev.type])
                if hasattr(ev , 'label') and ev.label is not None:
                    print '#' , ev.label, '#'
            
        for s,spiketr in enumerate(seg.get_spiketrains()) :
            print 'spiketrain' , s
            ts = spiketr.spike_times
            ax3.plot( ts , ones_like(ts)*s ,
                        linestyle = '',
                        marker = '|' ,
                        markersize = 5,
                        color = colors[s%7],
                        )
            
            if spiketr.get_spikes() is not None :
                for spike in spiketr.get_spikes():
                    if spike.waveform is not None:
                        tv = arange(spike.waveform.size , dtype = 'f')/spiketr.freq+spike.time
                        ax4.plot(tv,spike.waveform ,color = colors[s%7],)

                
        pylab.show()


if __name__ == "__main__":
    unittest.main()


