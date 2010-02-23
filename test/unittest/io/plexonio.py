# -*- coding: utf-8 -*-


import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import PlexonIO
from neo.core import *
from numpy import *
from scipy import rand
import pylab


class PlexonIOTest(unittest.TestCase):
    def testOpenFile1(self):
        #~ io = PlexonIO(filename = 'datafiles/File_plexon_1.plx')
        #~ io = PlexonIO(filename = 'datafiles/File_plexon_2.plx' )
        io = PlexonIO(filename = 'datafiles/File_plexon_3.plx')
        
        seg = io.read_segment(  load_spike_waveform = True)
        #~ seg = io.read_segment(  load_spike_waveform = False)
        
        fig = pylab.figure()
        ax = fig.add_subplot(4,1,1)
        ax2 = fig.add_subplot(4,1,2, sharex =ax)
        ax3 = fig.add_subplot(4,1,3, sharex =ax)
        ax4 = fig.add_subplot(4,1,4, sharex =ax)
        #~ print len(seg.get_analogsignals())
        #assert len(seg.get_analogsignals()) ==1
        colors = [  'b' , 'r' ,'g' , 'y' , 'k' , 'm' , 'c']
        for s,sig in enumerate(seg.get_analogsignals()):
            #~ print sig.signal.shape[0], sig.freq
            #assert sig.name == 'ImRK01G20'
            #assert sig.unit == 'pA'
            #assert sig.signal.shape[0] == 1912832
            ax.plot(sig.t() , sig.signal , color = colors[s%7] )
        
        #print len (seg.get_events() )
        #assert len (seg.get_events() )==0
        dict_color = { }
        for ev in seg.get_events():
            #~ print 'ev',ev.time
            if hasattr(ev , 'type'):
                if ev.type not in dict_color.keys():
                    dict_color[ev.type] = colors[len(dict_color) % 7]
                if hasattr(ev , 'waveform'):
                    tv = arange(ev.waveform.size , dtype = 'f')/ev.freq+ev.time
                    ax2.plot(tv,ev.waveform , color = dict_color[ev.type])
                    ax2.axvline(ev.time , color = dict_color[ev.type])
                else :
                    ax2.axvline(ev.time, color = dict_color[ev.type])
            else :
                ax2.axvline(ev.time, color = 'b')
                if hasattr(ev , 'label') and ev.label is not None:
                    pass
                    #~ print '#' , ev.label, '#'
        
        #~ print seg.get_epochs()
        for ep in seg.get_epochs():
            ax2.fill_betweenx([-100, 100], [ep.time , ep.time ], ep.time+ep.duration , alpha = .2)
            #~ print 'ep',  ep.time, ep.duration
            
        fig = pylab.figure()
        ax5 = fig.add_subplot(1,1,1)
        
        for s,spiketr in enumerate(seg.get_spiketrains()) :
            #~ print 'spiketrain' , s
            ts = spiketr.spike_times
            #~ print ts.shape
            ax3.plot( ts , ones_like(ts)*s ,
                        linestyle = '',
                        marker = '|' ,
                        markersize = 5,
                        color = colors[s%7],
                        )
            if spiketr._spikes is not None :
                for sp in  spiketr.get_spikes()[:200]:
                    vect_t  = arange(sp.waveform.size , dtype = 'f')/sp.freq
                    for w in range(sp.waveform.shape[0]) :
                        ax5.plot(vect_t , sp.waveform[w,:] , color = colors[s % 7] )
                        
        
        pylab.show()




if __name__ == "__main__":
    unittest.main()
    

