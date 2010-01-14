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
        #seg = spike2.read_segment( filename = 'datafiles/File_spike2_1.smr',transform_event_to_spike = ' 3 ')
        seg = spike2.read_segment( filename = 'datafiles/R05-C05C.SMR',transform_event_to_spike = [3 ])
#        seg = spike2.read_segment( filename = 'datafiles/R12-C10b.smr',)
#        seg = spike2.read_segment( filename = 'datafiles/R12-C11f.smr',)
#        seg = spike2.read_segment( filename = 'datafiles/R14-C02C.SMR',)
#        seg = spike2.read_segment( filename = 'datafiles/R21-C11d.smr',)
        #seg = spike2.read_segment( filename = 'datafiles/R25-C07C.SMR', transform_event_to_spike = [3 ])
#        seg = spike2.read_segment( filename = 'datafiles/J0_G3S2.SMR',)
        #seg = spike2.read_segment( filename = 'datafiles/example.smr',)
        #seg = spike2.read_segment( filename = 'datafiles/20091007_000.smr',)
        #seg = spike2.read_segment( filename = 'datafiles/20091103_000.smr',)
        #seg = spike2.read_segment( filename = 'datafiles/20091104_000.smr',)
        
        #seg = spike2.read_segment( filename = 'datafiles/data spike2 4channels 1 hypno 1textmarkchannel 1markerchannel.smr',)


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
                ax2.plot(tv,ev.waveform , color = dict_color[ev.type])
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
            
            for spike in spiketr.get_spikes():
                if spike.waveform is not None:
                    tv = arange(spike.waveform.size , dtype = 'f')/spiketr.freq+spike.time
                    ax4.plot(tv,spike.waveform ,color = colors[s%7],)

                
        pylab.show()


if __name__ == "__main__":
    unittest.main()


