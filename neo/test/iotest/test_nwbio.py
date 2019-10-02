#
"""
Tests of neo.io.nwbio
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import unittest
from neo.io.nwbio import NWBIO
from neo.test.iotest.common_io_test import BaseTestIO
import pynwb
from pynwb import *

from neo.core import AnalogSignal, SpikeTrain, Event, Epoch, IrregularlySampledSignal, Segment, Unit, Block
import quantities as pq
import numpy as np

# allensdk package
import allensdk
from allensdk import *
from pynwb import load_namespaces
from allensdk.brain_observatory.nwb.metadata import load_LabMetaData_extension
from allensdk.brain_observatory.behavior.schemas import OphysBehaviorMetaDataSchema, OphysBehaviorTaskParametersSchema
load_LabMetaData_extension(OphysBehaviorMetaDataSchema, 'AIBS_ophys_behavior')
load_LabMetaData_extension(OphysBehaviorTaskParametersSchema, 'AIBS_ophys_behavior')

class TestNWBIO(unittest.TestCase, ):
    ioclass = NWBIO
    files_to_download =  [
        # My NWB files
#              '/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_bis.nwb', # File created with the latest version of pynwb=1.0.1 only with ephys data File on my github page
#               '/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data_1_timestamp.nwb',
        # Files from Allen Institute
        # NWB files downloadable from http://download.alleninstitute.org/informatics-archive/prerelease/
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb'
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb'
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb'
              '/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb'
#              '/home/elodie/NWB_Files/NWB_org/behavior_ophys_session_775614751.nwb'
#              '/home/elodie/NWB_Files/NWB_org/ecephys_session_785402239.nwb'
    ]
    entities_to_test = files_to_download

    def test_read_analogsignal(self):
        sig_neo = AnalogSignal(signal=[.01, 3.3, 9.3], units='uV', sampling_rate=1*pq.Hz)
        self.assertTrue(isinstance(sig_neo, AnalogSignal))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        obj_nwb = r._handle_timeseries(False, 'name', 1)
        self.assertTrue(isinstance(obj_nwb, AnalogSignal))
        self.assertEqual(isinstance(obj_nwb, AnalogSignal), isinstance(sig_neo, AnalogSignal))
        self.assertTrue(obj_nwb.shape, sig_neo.shape)
        self.assertTrue(obj_nwb.sampling_rate, sig_neo.sampling_rate)
        self.assertTrue(obj_nwb.units, sig_neo.units)
        self.assertIsNotNone(obj_nwb, sig_neo)

    def test_read_irregularlysampledsignal(self, **kargs):
        irsig0 = IrregularlySampledSignal([0.0, 1.23, 6.78], [1, 2, 3], units='mV', time_units='ms')
        irsig1 = IrregularlySampledSignal([0.01, 0.03, 0.12]*pq.s, [[4, 5], [5, 4], [6, 3]]*pq.nA)
        self.assertTrue(isinstance(irsig0, IrregularlySampledSignal))
        self.assertTrue(isinstance(irsig1, IrregularlySampledSignal))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        irsig_nwb = r._handle_epochs_group(False, 'name')
        self.assertTrue(irsig_nwb, IrregularlySampledSignal)
        self.assertTrue(irsig_nwb, irsig0)
        self.assertTrue(irsig_nwb, irsig1)

    def test_read_event(self, **kargs):
        evt_neo = Event(np.arange(0, 30, 10)*pq.s, labels=np.array(['trig0', 'trig1', 'trig2'], dtype='S'))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        event_nwb = r._handle_epochs_group(False, 'name')
        self.assertTrue(event_nwb, evt_neo)
        self.assertIsNotNone(event_nwb, evt_neo)

    def test_read_epoch(self, **kargs):
        epc_neo = Epoch(times=np.arange(0, 30, 10)*pq.s,
                    durations=[10, 5, 7]*pq.ms,
                    labels=np.array(['btn0', 'btn1', 'btn2'], dtype='S'))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        epoch_nwb = r._handle_epochs_group(False, 'name')
        self.assertTrue(epoch_nwb, Epoch)
        self.assertTrue(epoch_nwb, epc_neo)
        self.assertIsNotNone(epoch_nwb, epc_neo)

    def test_read_segment(self, **kargs):
        seg = Segment(index=5)
        train0_neo = SpikeTrain(times=[.01, 3.3, 9.3], units='sec', t_stop=10)
        seg.spiketrains.append(train0_neo)
        sig0_neo = AnalogSignal(signal=[.01, 3.3, 9.3], units='uV', sampling_rate=1*pq.Hz)
        seg.analogsignals.append(sig0_neo)
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        seg_nwb = r._handle_epochs_group(False, 'name')
        self.assertTrue(seg, Segment)
        self.assertTrue(seg_nwb, Segment)
        self.assertTrue(seg_nwb, seg)
        self.assertIsNotNone(seg_nwb, seg)

    def test(self):
        # Spiketrain
        train = SpikeTrain([3, 4, 5] * pq.s, t_stop=10.0)
        unit = Unit()
        train.unit = unit
        unit.spiketrains.append(train)

        epoch = Epoch(np.array([0, 10, 20]),
                      np.array([2, 2, 2]),
                      np.array(["a", "b", "c"]),
                      units="ms")
        blk = Block()
        seg = Segment()
        seg.spiketrains.append(train)
        seg.epochs.append(epoch)
        epoch.segment = seg
        blk.segments.append(seg)
        r = NWBIO(filename=self.files_to_download[0] ,mode='r')

        r_blk = r.read_block()
        r_seg = r_blk.segments

    def test_read_block(self, filename=None):
        '''
        Test function to read neo block.
        '''
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        bl = r.read_block()

#    def test_write_segment(self, filename=None):
#        '''
#        Test function to write a segment.
#        '''
#        r = NWBIO(filename=self.files_to_download[0], mode='r')
#        ws = r._write_segment(None)


if __name__ == "__main__":
    print("pynwb.__version__ = ", pynwb.__version__)
    unittest.main()