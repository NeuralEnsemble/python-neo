#
"""
Tests of neo.io.nwbio
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import unittest
from neo.io.nwbio import NWBIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.core import AnalogSignal, SpikeTrain, Event, Epoch, IrregularlySampledSignal, Segment, Unit, Block, ChannelIndex, ImageSequence
import pynwb
from pynwb import *
import quantities as pq
import numpy as np

class TestNWBIO(unittest.TestCase, ):
    ioclass = NWBIO
    files_to_download =  [
#        Files from Allen Institute :
#        NWB files downloadable from http://download.alleninstitute.org/informatics-archive/prerelease/
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb'
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-3.nwb'
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-4.nwb'
#              '/home/elodie/NWB_Files/NWB_org/H19.29.141.11.21.01.nwb'
#        File created from Neo (Jupyter notebook "test_nwbio_class_from_Neo.ipynb")
###              '/home/elodie/env_NWB_py3/my_notebook/My_first_dataset_neo9.nwb'
              '/home/elodie/env_NWB_py3/my_notebook/My_first_dataset_neo10.nwb'
    ]
    entities_to_test = files_to_download

    def test_nwbio(self):
        reader = NWBIO(filename=self.files_to_download[0], mode='r')
        reader.read()

    def test_segment(self, **kargs):
        seg = Segment(index=5)
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        seg_nwb = r.read() # equivalent to read_all_blocks()
        self.assertTrue(seg, Segment)
        self.assertTrue(seg_nwb, Segment)
        self.assertTrue(seg_nwb, seg)
        self.assertIsNotNone(seg_nwb, seg)
        seg_nwb_one_block = r.read_block() # only for the first block
        self.assertTrue(seg_nwb_one_block, Segment)
        self.assertTrue(seg_nwb_one_block, seg)
        self.assertIsNotNone(seg_nwb_one_block, seg)

    def test_analogsignals_neo(self, **kargs):
        sig_neo = AnalogSignal(signal=[1, 2, 3], units='V', t_start=np.array(3.0)*pq.s, sampling_rate=1*pq.Hz)
        self.assertTrue(isinstance(sig_neo, AnalogSignal))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        obj_nwb = r.read()
        self.assertTrue(obj_nwb, AnalogSignal)
        self.assertTrue(obj_nwb, sig_neo)

    def test_ImageSequence_neo(self, **kargs):
        img_sequence_array = [[[column for column in range(2)]for row in range(3)] for frame in range(4)]
        image_neo = ImageSequence(img_sequence_array, units='V', sampling_rate=1*pq.Hz, spatial_scale=1*pq.micrometer)
        self.assertTrue(isinstance(image_neo, ImageSequence))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        obj_nwb = r.read()
        self.assertTrue(obj_nwb, ImageSequence)
        self.assertTrue(obj_nwb, image_neo)

    def test_calcium_imaging_data_neo(self, **kargs):
        img_sequence_array = [[[column for column in range(2)]for row in range(3)] for frame in range(4)]
        calcium_imaging_data_neo = ImageSequence(img_sequence_array, units='V', sampling_rate=1*pq.Hz, spatial_scale=1*pq.micrometer)
        self.assertTrue(isinstance(calcium_imaging_data_neo, ImageSequence))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        cid_nwb = r.read()
        self.assertTrue(cid_nwb, ImageSequence)
        self.assertTrue(cid_nwb, calcium_imaging_data_neo)

    def test_spiketrains_neo(self, **kargs):
        train = SpikeTrain(times=[1, 2, 3]*pq.s, t_start=1.0, t_stop=10.0) 
        self.assertTrue(isinstance(train, SpikeTrain))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        obj_nwb = r.read()
        self.assertTrue(obj_nwb, SpikeTrain)
        self.assertTrue(obj_nwb, train)

    def test_epochs_neo(self, **kargs):
        epc = Epoch(times=np.arange(0, 30, 10)*pq.s,
                    durations=[10, 5, 7]*pq.ms,
                    labels=np.array(['btn0', 'btn1', 'btn2'], dtype='S'))
        self.assertTrue(isinstance(epc, Epoch))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        obj_nwb = r.read()
        self.assertTrue(obj_nwb, Epoch)
        self.assertTrue(obj_nwb, epc)

    def test_read_irregularlysampledsignal(self, **kargs):
        irsig0 = IrregularlySampledSignal([0.0, 1.23, 6.78], [1, 2, 3], units='mV', time_units='ms')
        irsig1 = IrregularlySampledSignal([0.01, 0.03, 0.12]*pq.s, [[4, 5], [5, 4], [6, 3]]*pq.nA)
        self.assertTrue(isinstance(irsig0, IrregularlySampledSignal))
        self.assertTrue(isinstance(irsig1, IrregularlySampledSignal))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        irsig_nwb = r.read()
        self.assertTrue(irsig_nwb, IrregularlySampledSignal)
        self.assertTrue(irsig_nwb, irsig0)
        self.assertTrue(irsig_nwb, irsig1)

    def test_read_event(self, **kargs):
        evt_neo = Event(np.arange(0, 30, 10)*pq.s, labels=np.array(['trig0', 'trig1', 'trig2'], dtype='S'))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        event_nwb = r.read()
        self.assertTrue(event_nwb, evt_neo)
        self.assertIsNotNone(event_nwb, evt_neo)

    def test_read_epoch(self, **kargs):
        epc_neo = Epoch(times=np.arange(0, 30, 10)*pq.s,
                    durations=[10, 5, 7]*pq.ms,
                    labels=np.array(['btn0', 'btn1', 'btn2'], dtype='S'))
        r = NWBIO(filename=self.files_to_download[0], mode='r')
        epoch_nwb = r.read()
        self.assertTrue(epoch_nwb, Epoch)
        self.assertTrue(epoch_nwb, epc_neo)
        self.assertIsNotNone(epoch_nwb, epc_neo)

if __name__ == "__main__":
    print("pynwb.__version__ = ", pynwb.__version__)
    unittest.main()