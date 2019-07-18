
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
from neo.core import AnalogSignal, SpikeTrain, Event, Epoch, IrregularlySampledSignal, Segment
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
        print("--- Test AnalogSignal ---")
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

    def test_read_irregularlysampledsignal(self, **kargs):
        print("--- Test IrregularlySampledSignal ---")
        irsig0 = IrregularlySampledSignal([0.0, 1.23, 6.78], [1, 2, 3], units='mV', time_units='ms')
        #print("irsig0 = ", irsig0)
        irsig1 = IrregularlySampledSignal([0.01, 0.03, 0.12]*pq.s, [[4, 5], [5, 4], [6, 3]]*pq.nA)
        #print("irsig1 = ", irsig1)
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
        #print("irsig_nwb = ", irsig_nwb)
        self.assertTrue(irsig_nwb, IrregularlySampledSignal)
        self.assertTrue(irsig_nwb, irsig0)
        self.assertTrue(irsig_nwb, irsig1)

    def test_read_spiketrain(self, **kargs):
        print("--- Test spiketrain ---")
        train_neo = SpikeTrain([3, 4, 5]*pq.s, t_stop=10.0)
        #print("train_neo = ", train_neo)
        self.assertTrue(isinstance(train_neo, SpikeTrain))

        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
        # Files from Allen Institute
#        r = NWBIO('/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        train_nwb = r._handle_acquisition_group(False, 1)
        #print("train_nwb = ", train_nwb)
        self.assertTrue(isinstance(train_nwb, SpikeTrain))
        self.assertEqual(isinstance(train_nwb, SpikeTrain), isinstance(train_neo, SpikeTrain))
        self.assertTrue(train_nwb.shape, train_neo.shape)
        self.assertTrue(train_nwb.sampling_rate, train_neo.sampling_rate)
        self.assertTrue(train_nwb.units, train_neo.units)
        self.assertIsNotNone(train_nwb, train_neo)

    def test_read_event(self, **kargs):
        print("--- Test Event ---")
        evt_neo = Event(np.arange(0, 30, 10)*pq.s, labels=np.array(['trig0', 'trig1', 'trig2'], dtype='S'))
        #print("evt_neo = ", evt_neo)

        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
        # Files from Allen Institute
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        event_nwb = r._handle_epochs_group(False, 'name')
        #print("event_nwb = ", event_nwb)
        self.assertTrue(event_nwb, evt_neo)
        self.assertIsNotNone(event_nwb, evt_neo)
    
    def test_read_epoch(self, **kargs):
        print("--- Test Epoch ---")
        epc_neo = Epoch(times=np.arange(0, 30, 10)*pq.s,
                    durations=[10, 5, 7]*pq.ms,
                    labels=np.array(['btn0', 'btn1', 'btn2'], dtype='S'))
        #print("epc_neo = ", epc_neo)

        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
        # Files from Allen Institute
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        epoch_nwb = r._handle_epochs_group(False, 'name')
        #print("epoch_nwb = ", epoch_nwb)
        self.assertTrue(epoch_nwb, Epoch)
        self.assertTrue(epoch_nwb, epc_neo)
        self.assertIsNotNone(epoch_nwb, epc_neo)

    def test_read_segment(self, **kargs):
        print("--- Test Segment ---")
        seg = Segment(index=5)
        #print("seg = ", seg)
        train0_neo = SpikeTrain(times=[.01, 3.3, 9.3], units='sec', t_stop=10)
        #print("train0_neo = ", train0_neo)
        seg.spiketrains.append(train0_neo)
        #print("seg.spiketrains.append(train0_neo) = ", seg.spiketrains.append(train0_neo))
        sig0_neo = AnalogSignal(signal=[.01, 3.3, 9.3], units='uV', sampling_rate=1*pq.Hz)
        #print("sig0_neo = ", sig0_neo)
        seg.analogsignals.append(sig0_neo)
        #print("seg.analogsignals.append(sig0_neo) = ", seg.analogsignals.append(sig0_neo))

        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
        # Files from Allen Institute
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        seg_nwb = r._handle_epochs_group(False, 'name')
        #print("seg_nwb = ", seg_nwb)
        self.assertTrue(seg, Segment)
        self.assertTrue(seg_nwb, Segment)
        self.assertTrue(seg_nwb, seg)
        self.assertIsNotNone(seg_nwb, seg)

    def test_read_block(self, filename=None):
        '''
        Test function to read neo block.
        '''
        print("*** def test_read_block ***")
        # Files to test
        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb')
        # Files from Allen Institute
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb')
#        r = NWBIO(filename='/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb')
        #print("-----------------------r = ", r)
        bl = r.read_block()
        #print("bl = ", bl)
        print("*** End ***")
        

if __name__ == "__main__":
    print("pynwb.__version__ = ", pynwb.__version__)
    unittest.main()



