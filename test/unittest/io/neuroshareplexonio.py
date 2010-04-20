# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import NeurosharePlexonIO
from neo.core import *
from numpy import *
from scipy import rand
import pylab

class NeurosharePlexonIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        io = NeurosharePlexonIO(filename = 'datafiles/test.nex',)
        seg = io.read_segment( )
        
        #seg = plexon.read_segment( filename = 'datafiles/TestDataFile1.nex',)
        #seg = plexon.read_segment( filename = 'datafiles/TestDataFileForNeuroshare.nex',)


        fig = pylab.figure()
        ax = fig.add_subplot(4,1,1)
        ax2 = fig.add_subplot(4,1,2, sharex =ax)
        ax3 = fig.add_subplot(4,1,3, sharex =ax)
        ax4 = fig.add_subplot(4,1,4, sharex =ax)
        print len(seg.get_analogsignals())
        #assert len(seg.get_analogsignals()) ==1
        colors = [  'b' , 'r' ,'g' , 'y' , 'k' , 'm' , 'c']
        for s,sig in enumerate(seg.get_analogsignals()):
            #print sig.signal.shape[0], sig.sampling_rate
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
                tv = arange(ev.waveform.size , dtype = 'f')/ev.sampling_rate+ev.time
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
                        tv = arange(spike.waveform.size , dtype = 'f')/spiketr.sampling_rate+spike.time
                        ax4.plot(tv,spike.waveform ,color = colors[s%7],)

                
        pylab.show()


if __name__ == "__main__":
    unittest.main()


