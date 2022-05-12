"""
Tests of neo.io.edfio
"""

import unittest
import numpy as np
import quantities as pq

from neo.io.edfio import EDFIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.io.proxyobjects import AnalogSignalProxy
from neo import AnalogSignal


class TestEDFIO(BaseTestIO, unittest.TestCase, ):
    ioclass = EDFIO
    entities_to_download = ['edf']
    entities_to_test = [
        'edf/edf+C.edf',
    ]

    def setUp(self):
        super().setUp()
        self.filename = self.get_local_path('edf/edf+C.edf')

    def test_read_block(self):
        """
        Test reading the complete block and general annotations
        """
        with EDFIO(self.filename) as io:
            bl = io.read_block()
            self.assertTrue(bl.annotations)

            seg = bl.segments[0]
            assert seg.name == 'Seg #0 Block #0'
            for anasig in seg.analogsignals:
                assert anasig.name is not None

    def test_read_segment_with_time_slice(self):
        """
        Test loading of a time slice and check resulting times
        """
        with EDFIO(self.filename) as io:
            seg = io.read_segment(time_slice=None)

            # data file does not contain spike, event or epoch timestamps
            self.assertEqual(len(seg.spiketrains), 0)
            self.assertEqual(len(seg.events), 1)
            self.assertEqual(len(seg.events[0]), 0)
            self.assertEqual(len(seg.epochs), 1)
            self.assertEqual(len(seg.epochs[0]), 0)
            for asig in seg.analogsignals:
                self.assertEqual(asig.shape[0], 256)
            n_channels = sum(a.shape[-1] for a in seg.analogsignals)
            self.assertEqual(n_channels, 5)

            t_start, t_stop = 500 * pq.ms, 800 * pq.ms
            seg = io.read_segment(time_slice=(t_start, t_stop))

            self.assertAlmostEqual(seg.t_start.rescale(t_start.units), t_start, delta=5.)
            self.assertAlmostEqual(seg.t_stop.rescale(t_stop.units), t_stop, delta=5.)

    def test_compare_data(self):
        """
        Compare data from AnalogSignal with plain data stored in text file
        """
        with EDFIO(self.filename) as io:
            plain_data = np.loadtxt(io.filename.replace('.edf', '.txt'), dtype=np.int16)
            seg = io.read_segment(lazy=True)

            anasigs = seg.analogsignals
            self.assertEqual(len(anasigs), 5)  # all channels have different units, so expecting 5
            for aidx, anasig in enumerate(anasigs):
                # comparing raw data to original values
                ana_data = anasig.load(magnitude_mode='raw')
                np.testing.assert_array_equal(ana_data.magnitude, plain_data[:, aidx:aidx + 1])

                # comparing floating data to original values * gain factor
                ch_head = io.edf_reader.getSignalHeader(aidx)
                physical_range = ch_head['physical_max'] - ch_head['physical_min']
                # number of digital values used (+1 to account for '0' value)
                digital_range = ch_head['digital_max'] - ch_head['digital_min'] + 1

                gain = physical_range / digital_range
                ana_data = anasig.load(magnitude_mode='rescaled')
                rescaled_data = plain_data[:, aidx:aidx + 1] * gain
                np.testing.assert_array_equal(ana_data.magnitude, rescaled_data)


if __name__ == "__main__":
    unittest.main()
