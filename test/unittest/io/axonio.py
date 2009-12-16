# -*- coding: utf-8 -*-



import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))

from neo.io import axonio
from neo.core import *
from numpy import *
from scipy import rand
import pylab


class AxonIOTest(unittest.TestCase):
    
    def testOpenFile1(self):
        axon = axonio.AxonIO()
#        block = axon.read_block( filename = 'datafiles/File_axon_1.abf',)
#        
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/gap free/05125006.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/gap free/05125007.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/gap free/Axo05611_0000.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/gap free/Axo05611_0002.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/gap free/Axo05611_0003.abf',)
#        
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_CClamp/05611001_stim_nerf.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_CClamp/05611003_creneau.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_CClamp/05611005_stim_nerf.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_CClamp/05611006_stim_nerf.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_CClamp/05611009_creneau.abf',)
#        
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_VClamp/05125008_creneau.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_VClamp/05426021_rampe.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_VClamp/05426022_rampe.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_VClamp/05426023_rampe.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_VClamp/05511002_creneau.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_VClamp/05511011_creneau.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_VClamp/05611004_ncreneaux.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon9/waveform_VClamp/05611010_ncreneaux.abf',)
#        
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon10/gap_free_CClamp/07502000.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon10/gap_free_CClamp/07502001.abf',)
#        
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon10/gap_free_VClamp/08407002.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon10/gap_free_VClamp/08407003.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon10/gap_free_VClamp/08407004.abf',)
        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon10/gap_free_VClamp/09512005_tag.abf',)
#        block = axon.read_block( filename = 'datafiles/ABF/fichiers abf axon10/gap_free_VClamp/09512008_tag.abf',)
        
        
       
        for seg in block.get_segments()[:5] :
            print seg
            fig = pylab.figure()
            ax = fig.add_subplot(1,1,1)
            #assert len(seg.get_analogsignals()) ==7
            for sig in seg.get_analogsignals():
                print sig.signal.shape[0], sig.name, sig.num, sig.unit, sig.freq
                
                #assert sig.signal.shape[0] == 240000
                ax.plot(sig.t(),sig.signal)
                #print sig.num, sig.label , sig.ground
            
            print len (seg.get_events() )
            #assert len (seg.get_events() )==47 
            for ev in seg.get_events():
                #print ev.time
                ax.axvline(ev.time)
        
        pylab.show()








if __name__ == "__main__":
    unittest.main()
    

