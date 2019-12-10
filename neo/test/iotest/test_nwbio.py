#
"""
Tests of neo.io.nwbio
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import unittest
from neo.io.nwbio import NWBIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.core import AnalogSignal, SpikeTrain, Event, Epoch, IrregularlySampledSignal, Segment, Unit, Block, ChannelIndex
import pynwb
from pynwb import *
import quantities as pq
import numpy as np

class TestNWBIO(unittest.TestCase, ):
    ioclass = NWBIO
    files_to_download =  [
        # My NWB files
#              '/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_bis.nwb', # File created with the latest version of pynwb=1.0.1 only with ephys data File on my github page
###              '/Users/legouee/NWBwork/my_notebook/NWB_File_python_3_pynwb_101_ephys_data_bis.nwb'
#              '/Users/legouee/NWBwork/my_notebook/My_first_dataset.nwb'
              '/Users/legouee/NWBwork/my_notebook/My_first_dataset_neo8.nwb'

        # Files from Allen Institute
        # NWB files downloadable from http://download.alleninstitute.org/informatics-archive/prerelease/
###              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb'
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb'
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb'
###              '/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb'
#              '/home/elodie/NWB_Files/NWB_org/behavior_ophys_session_775614751.nwb'
#              '/home/elodie/NWB_Files/NWB_org/ecephys_session_785402239.nwb'

        # File written with NWBIO class()
###              '/home/elodie/env_NWB_py3/my_notebook/my_first_test_neo_to_nwb.nwb'
###              '/home/elodie/env_NWB_py3/my_notebook/my_first_test_neo_to_nwb_test_NWBIO.nwb'
#              '/home/elodie/env_NWB_py3/my_notebook/my_first_test_neo_to_nwb_test_NWBIO_2.nwb'
###            '/home/elodie/env_NWB_py3/my_notebook/my_first_test_neo_to_nwb_test_NWBIO.nwb'
    ]
    entities_to_test = files_to_download


    def test_nwbio(self):
        # read the blocks
        reader = NWBIO(filename=self.files_to_download[0], mode='r')
        print("reader = ", reader)
#        print("reader.read() = ", reader.read())
        
        print("reader.read_block() = ", reader.read_block())
        print("   ")
#        blocks = reader.read(lazy=False)

        #-------------------------------------------------------
        blocks=[]
        for ind in range(2): # 2 blocks
            blk = Block(name='%s' %ind)
            blocks.append(blk)
        #-------------------------------------------------------

        # access to segments
        for block in blocks:
            # Tests of Block
            self.assertTrue(isinstance(block.name, str))
            # Segment
            for segment in block.segments:
                self.assertEqual(segment.block, block)
                # AnalogSignal
                for asig in segment.analogsignals:
                    self.assertTrue(isinstance(asig, AnalogSignal))
                    self.assertTrue(asig.sampling_rate, pq.Hz)
                    self.assertTrue(asig.units, pq)
                # Spiketrain
                for st in segment.spiketrains:
                    self.assertTrue(isinstance(st, SpikeTrain))          


    def test_segment(self, **kargs):
        seg = Segment(index=5)
        r = NWBIO(filename=self.files_to_download[0], mode='r')


#        #-------------------------------------------------------
#        blocks=[]
#        for ind in range(2): # 2 blocks
#            blk = Block(name='%s' %ind)
#            blocks.append(blk)
#        #-------------------------------------------------------
#        seg_nwb = r.read()
##        seg_nwb = r.read(blocks) # equivalent to read_all_blocks()


        seg_nwb = r.read_block()        
        self.assertTrue(seg, Segment)
        self.assertTrue(seg_nwb, Segment)
        self.assertTrue(seg_nwb, seg)
        self.assertIsNotNone(seg_nwb, seg)

    def test_analogsignals_neo(self, **kargs):
        sig_neo = AnalogSignal(signal=[1, 2, 3], units='V', t_start=np.array(3.0)*pq.s, sampling_rate=1*pq.Hz)
        self.assertTrue(isinstance(sig_neo, AnalogSignal))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
#        obj_nwb = r.read()
        obj_nwb = r.read_block()        
        self.assertTrue(obj_nwb, AnalogSignal)
        self.assertTrue(obj_nwb, sig_neo)

    
    def test_read_irregularlysampledsignal(self, **kargs):
        irsig0 = IrregularlySampledSignal([0.0, 1.23, 6.78], [1, 2, 3], units='mV', time_units='ms')
        irsig1 = IrregularlySampledSignal([0.01, 0.03, 0.12]*pq.s, [[4, 5], [5, 4], [6, 3]]*pq.nA)
        self.assertTrue(isinstance(irsig0, IrregularlySampledSignal))
        self.assertTrue(isinstance(irsig1, IrregularlySampledSignal))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
#        irsig_nwb = r.read()
        irsig_nwb = r.read_block()        
        self.assertTrue(irsig_nwb, IrregularlySampledSignal)
        self.assertTrue(irsig_nwb, irsig0)
        self.assertTrue(irsig_nwb, irsig1)

    def test_read_event(self, **kargs):
        evt_neo = Event(np.arange(0, 30, 10)*pq.s, labels=np.array(['trig0', 'trig1', 'trig2'], dtype='S'))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
#        event_nwb = r.read()
        event_nwb = r.read_block()
        self.assertTrue(event_nwb, evt_neo)
        self.assertIsNotNone(event_nwb, evt_neo)

    def test_read_epoch(self, **kargs):
        epc_neo = Epoch(times=np.arange(0, 30, 10)*pq.s,
                    durations=[10, 5, 7]*pq.ms,
                    labels=np.array(['btn0', 'btn1', 'btn2'], dtype='S'))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
#        epoch_nwb = r.read()
        epoch_nwb = r.read_block()
        self.assertTrue(epoch_nwb, Epoch)
        self.assertTrue(epoch_nwb, epc_neo)
        self.assertIsNotNone(epoch_nwb, epc_neo)

    def test_write_NWB_Files(self):
        '''
        Test function to write several blocks containing several segments and analogsignals.
        '''
        print("Test function test_write_NWB_Files")
        blocks = []

        bl0 = Block(name='First block')
        bl1 = Block(name='Second block')
        bl2 = Block(name='Third block')
        print("bl0.segments = ", bl0.segments)      
        print("bl1.segments = ", bl1.segments)
        print("bl2.segments = ", bl2.segments)
        blocks = [bl0, bl1, bl2]
        print("blocks = ", blocks)

        num_seg = 3 # number of segments

        for blk in blocks:
            print("blk = ", blk)
            for ind in range(num_seg): # number of Segment
                seg = Segment(name='segment %d' % ind, index=ind)
                blk.segments.append(seg)

            for seg in blk.segments: # AnalogSignal objects
                # 3 AnalogSignals
                print("seg = ", seg)
                a = AnalogSignal(np.random.randn(num_seg, 44)*pq.nA, sampling_rate=10*pq.kHz)
                b = AnalogSignal(np.random.randn(num_seg, 64)*pq.nA, sampling_rate=10*pq.kHz)
                c = AnalogSignal(np.random.randn(num_seg, 33)*pq.nA, sampling_rate=10*pq.kHz)
       
                seg.analogsignals.append(a) 
                seg.analogsignals.append(b) 
                seg.analogsignals.append(c)

        print("END blocks = ", blocks)

        # Save the file
#        filename = '/home/elodie/env_NWB_py3/my_notebook/my_first_test_neo_to_nwb_test_NWBIO.nwb'
        filename = '/Users/legouee/NWBwork/my_notebook/my_first_test_neo_to_nwb_test_NWBIO_in_test_nwbio.nwb'
        print("filename = ", filename)
        w_file = NWBIO(filename=filename, mode='w') # Write the .nwb file
        print("w_file = ", w_file)
        blocks = w_file.write(blk)
        print("*** END test_write_NWB_Files ***")


if __name__ == "__main__":
    print("pynwb.__version__ = ", pynwb.__version__)
    unittest.main()