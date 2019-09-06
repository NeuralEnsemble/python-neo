
"""
Tests of neo.io.nwbio
"""

#from __future__ import division
#
#import sys
#import unittest
#try:
#    import unittest2 as unittest
#except ImportError:
#    import unittest
#try:
#    import pynwb
#    HAVE_NWB = True
#except ImportError:
#    HAVE_NWB = False
#from neo.io import NWBIO
#from neo.test.iotest.common_io_test import BaseTestIO

from __future__ import unicode_literals, print_function, division, absolute_import
import unittest
from neo.io.nwbio import NWBIO
from neo.test.iotest.common_io_test import BaseTestIO
import pynwb
from pynwb import *

# Tests
from neo.core import AnalogSignal, SpikeTrain, Event, Epoch, IrregularlySampledSignal, Segment, Unit, Block
import quantities as pq
import numpy as np

#@unittest.skipUnless(HAVE_NWB, "requires nwb")
#class TestNWBIO(BaseTestIO, unittest.TestCase, ):
class TestNWBIO(unittest.TestCase, ):
    ioclass = NWBIO

    files_to_test = ['/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb']
#    files_to_test = ['/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb']
    # Files from Allen Institute
#    files_to_test = ['/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb']
#    files_to_test = ['/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb']
#    files_to_test = ['/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb']
#    files_to_test = ['/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb']

    files_to_download =  [
              '/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb', # File created with the latest version of pynwb=1.0.1 only with ephys data File on my github
#               '/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb',
                # Files from Allen Institute
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb' # NWB file downloaded from http://download.alleninstitute.org/informatics-archive/prerelease/H19.28.012.11.05-2.nwb
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb'
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb'
#              '/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb'
    ]

    entities_to_test = files_to_download

    def test_read_analogsignal(self):
        sig_neo = AnalogSignal(signal=[.01, 3.3, 9.3], units='uV', sampling_rate=1*pq.Hz)
        self.assertTrue(isinstance(sig_neo, AnalogSignal))
        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
        # Files from Allen Institute
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        obj_nwb = r._handle_timeseries(False, 'name', 1)
        self.assertTrue(isinstance(obj_nwb, AnalogSignal))
        self.assertEqual(isinstance(obj_nwb, AnalogSignal), isinstance(sig_neo, AnalogSignal))
        self.assertTrue(obj_nwb.shape, sig_neo.shape)
        self.assertTrue(obj_nwb.sampling_rate, sig_neo.sampling_rate)
        self.assertTrue(obj_nwb.units, sig_neo.units)
        self.assertIsNotNone(obj_nwb, sig_neo)

################################################################ # Error _handle_epochs_group
    def test_read_irregularlysampledsignal(self, **kargs):
        irsig0 = IrregularlySampledSignal([0.0, 1.23, 6.78], [1, 2, 3], units='mV', time_units='ms')
        irsig1 = IrregularlySampledSignal([0.01, 0.03, 0.12]*pq.s, [[4, 5], [5, 4], [6, 3]]*pq.nA)
        self.assertTrue(isinstance(irsig0, IrregularlySampledSignal))
        self.assertTrue(isinstance(irsig1, IrregularlySampledSignal))

        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
        # Files from Allen Institute
#        r = NWBIO('/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = ('/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = ('/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = ('/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        irsig_nwb = r._handle_epochs_group(False, 'name')
        self.assertTrue(irsig_nwb, IrregularlySampledSignal)
        self.assertTrue(irsig_nwb, irsig0)
        self.assertTrue(irsig_nwb, irsig1)

    def test_read_spiketrain(self, **kargs):
        train_neo = SpikeTrain([3, 4, 5]*pq.s, t_stop=10.0)
        self.assertTrue(isinstance(train_neo, SpikeTrain))

        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')#
        # Files from Allen Institute
#        r = NWBIO('/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        train_nwb = r._handle_acquisition_group(False, 1)
        self.assertTrue(isinstance(train_nwb, SpikeTrain))
        self.assertEqual(isinstance(train_nwb, SpikeTrain), isinstance(train_neo, SpikeTrain))
        self.assertTrue(train_nwb.shape, train_neo.shape)
        self.assertTrue(train_nwb.sampling_rate, train_neo.sampling_rate)
        self.assertTrue(train_nwb.units, train_neo.units)
        self.assertIsNotNone(train_nwb, train_neo)

    def test_read_event(self, **kargs):
        evt_neo = Event(np.arange(0, 30, 10)*pq.s, labels=np.array(['trig0', 'trig1', 'trig2'], dtype='S'))

        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
        # Files from Allen Institute
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        event_nwb = r._handle_epochs_group(False, 'name')
        self.assertTrue(event_nwb, evt_neo)
        self.assertIsNotNone(event_nwb, evt_neo)

    def test_read_epoch(self, **kargs):
        epc_neo = Epoch(times=np.arange(0, 30, 10)*pq.s,
                    durations=[10, 5, 7]*pq.ms,
                    labels=np.array(['btn0', 'btn1', 'btn2'], dtype='S'))

        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
        # Files from Allen Institute
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        epoch_nwb = r._handle_epochs_group(False, 'name')
        self.assertTrue(epoch_nwb, Epoch)
        self.assertTrue(epoch_nwb, epc_neo)
        self.assertIsNotNone(epoch_nwb, epc_neo)

#    def test_read_segment(self, **kargs):
#        print("--- Test Segment ---")
#        seg = Segment(index=5)
#        print("seg = ", seg)
#        train0_neo = SpikeTrain(times=[.01, 3.3, 9.3], units='sec', t_stop=10)
#        #print("train0_neo = ", train0_neo)
#        seg.spiketrains.append(train0_neo)
#        #print("seg.spiketrains.append(train0_neo) = ", seg.spiketrains.append(train0_neo))
#        sig0_neo = AnalogSignal(signal=[.01, 3.3, 9.3], units='uV', sampling_rate=1*pq.Hz)
#        #print("sig0_neo = ", sig0_neo)
#        seg.analogsignals.append(sig0_neo)
#        #print("seg.analogsignals.append(sig0_neo) = ", seg.analogsignals.append(sig0_neo))
#
#        # Files to test
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
##        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
#        # Files from Allen Institute
##        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
##        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
##        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
##        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
#        seg_nwb = r._handle_epochs_group(False, 'name')
#        print("seg_nwb = ", seg_nwb)
#        self.assertTrue(seg, Segment)
#        print("self.assertTrue(seg, Segment) = ", self.assertTrue(seg, Segment))
#        self.assertTrue(seg_nwb, Segment)
#        #print("self.assertTrue(seg_nwb, Segment) = ", self.assertTrue(seg_nwb, Segment))
#        self.assertTrue(seg_nwb, seg)
#        #print("self.assertTrue(seg_nwb, seg) = ", self.assertTrue(seg_nwb, seg))
#        self.assertIsNotNone(seg_nwb, seg)
#        #print("self.assertIsNotNone(seg_nwb, seg) = ", self.assertIsNotNone(seg_nwb, seg))
####        print("self.assertEqual(seg, Segment) = ", self.assertEqual(seg_nwb, Segment))
#        print("self.assertIsInstance(seg, Segment) = ", self.assertIsInstance(seg, Segment))
#    #    print("self.assertEqual(seg, Segment) = ", self.assertEqual(seg, Segment))
#       # print("self.assertIsInstance(seg_nwb, Segment) = ", self.assertIs(seg_nwb, Segment))
##        from neo.core import Unit
##        self.assertIsInstance(seg.spiketrains[0].unit, Unit)
#        print("self.assertIsNotNone(seg_nwb, Segment) = ", self.assertIsNotNone(seg_nwb, Segment))
#        print("self.assertIsNotNone(seg, Segment) = ", self.assertIsNotNone(seg, Segment))
#        print("self.assertIsNotNone(seg_nwb, seg) = ", self.assertIsNotNone(seg_nwb, seg))
#        print("self.assertNotIsInstance(seg_nwb, seg) = ", self.assertNotIsInstance(seg_nwb, Segment))



#    def test_read_segment_lazy(self):
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        print("r = ", r)
##        seg = r.read_segment(lazy=True)
#        seg = r._handle_epochs_group(True,'name')
#        print("seg = ", seg)
#        for ana in seg.analogsignals:
#            assert isinstance(ana, AnalogSignalProxy)
##            ana = ana.load()
##            assert isinstance(ana, AnalogSignal)#
#        for st in seg.spiketrains:
##            assert isinstance(st, SpikeTrainProxy)
##            st = st.load()
##            assert isinstance(st, SpikeTrain)



#    def test(self):
#
#        # Spiketrain
#        train = SpikeTrain([3, 4, 5] * pq.s, t_stop=10.0)
#        unit = Unit()
#        train.unit = unit
#        unit.spiketrains.append(train)
#
#        epoch = Epoch(np.array([0, 10, 20]),
#                      np.array([2, 2, 2]),
#                      np.array(["a", "b", "c"]),
#                      units="ms")
#
#        blk = Block()
#        seg = Segment()
#        seg.spiketrains.append(train)
#        seg.epochs.append(epoch)
#        epoch.segment = seg
#        blk.segments.append(seg)
#
#        reader = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        print("reader = ", reader)
#        r_blk = reader.read_block()
#        print("r_blk = ", r_blk)
##        r_seg = r_blk.segments[0]
#        r_seg = r_blk.segments
#        print("r_seg = ", r_seg)
##        self.assertIsInstance(r_seg.spiketrains[0].unit, Unit)
##        self.assertIsInstance(r_seg.epochs[0], Epoch)




    def test_read_block(self, filename=None):
        '''
        Test function to read neo block.
        '''
        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
        # Files from Allen Institute
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        bl = r.read_block()
        print("bl = ", bl)


    def test_write_segment(self, filename=None):
        '''
        Test function to write a segment.
        '''
        print("*** def test_write_segment ***")
        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
        # Files from Allen Institute
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        print("-----------------------r = ", r)
        ws = r._write_segment(None)
        print("ws = ", ws)


        

if __name__ == "__main__":
    print("pynwb.__version__ = ", pynwb.__version__)
    unittest.main()



