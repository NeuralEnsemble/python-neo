"""
Tests of neo.io.medio
"""

import pathlib
import unittest

import quantities as pq
import numpy as np

from neo.io.medio import MedIO
from neo.test.iotest.common_io_test import BaseTestIO

try:
    import dhn_med_py

    HAVE_DHN_MED = True
except ImportError:
    HAVE_DHN_MED = False


# This runs standard tests, this is mandatory for all IOs
@unittest.skipUnless(HAVE_DHN_MED, "requires dhn_med_py package and all its dependencies")
class TestMedIO(
    BaseTestIO,
    unittest.TestCase,
):
    ioclass = MedIO
    entities_to_download = ["med"]
    entities_to_test = ["med/sine_waves.medd", "med/test.medd"]

    def setUp(self):

        super().setUp()
        self.dirname = self.get_local_path("med/sine_waves.medd")
        self.dirname2 = self.get_local_path("med/test.medd")
        self.password = "L2_password"

    def test_read_segment_lazy(self):

        r = MedIO(self.dirname, self.password)
        seg = r.read_segment(lazy=False)

        # There will only be one analogsignal in this reading
        self.assertEqual(len(seg.analogsignals), 1)
        # Test that the correct number of samples are read, 5760000 samps for 3 channels
        self.assertEqual(seg.analogsignals[0].shape[0], 5760000)
        self.assertEqual(seg.analogsignals[0].shape[1], 3)

        # Test the first sample value of all 3 channels, which are
        # known to be [-1, -4, -4]
        np.testing.assert_array_equal(seg.analogsignals[0][0][:3], [-1, -4, -4])

        for anasig in seg.analogsignals:
            self.assertNotEqual(anasig.size, 0)
        for st in seg.spiketrains:
            self.assertNotEqual(st.size, 0)

        # annotations
        # assert 'seg_extra_info' in seg.annotations
        assert seg.name == "Seg #0 Block #0"
        for anasig in seg.analogsignals:
            assert anasig.name is not None
        for ev in seg.events:
            assert ev.name is not None
        for ep in seg.epochs:
            assert ep.name is not None

        r.close()

    def test_read_block(self):

        r = MedIO(self.dirname, self.password)
        bl = r.read_block(lazy=True)
        self.assertTrue(bl.annotations)

        for count, seg in enumerate(bl.segments):
            assert seg.name == "Seg #" + str(count) + " Block #0"

        for anasig in seg.analogsignals:
            assert anasig.name is not None

        # Verify that the block annotations from the MED session are
        # read properly.  There are a lot of annotations, so we'll just
        # spot-check a couple of them.
        assert bl.annotations["metadata"]["recording_country"] == "United States"
        assert bl.annotations["metadata"]["AC_line_frequency"] == 60.0

        r.close()

    def test_read_segment_with_time_slice(self):
        """
        Test loading of a time slice and check resulting times
        """
        r = MedIO(self.dirname, self.password)
        seg = r.read_segment(time_slice=None)

        # spike and epoch timestamps are not being read
        self.assertEqual(len(seg.spiketrains), 0)
        self.assertEqual(len(seg.epochs), 1)
        self.assertEqual(len(seg.epochs[0]), 0)

        # Test for 180 events (1 per second for 3 minute recording)
        self.assertEqual(len(seg.events), 1)
        self.assertEqual(len(seg.events[0]), 180)

        for asig in seg.analogsignals:
            self.assertEqual(asig.shape[0], 5760000)
        n_channels = sum(a.shape[-1] for a in seg.analogsignals)
        self.assertEqual(n_channels, 3)

        t_start, t_stop = 500 * pq.ms, 800 * pq.ms
        seg = r.read_segment(time_slice=(t_start, t_stop))

        # Test that 300 ms were read, which at 32 kHz, is 9600 samples
        self.assertAlmostEqual(seg.analogsignals[0].shape[0], 9600, delta=1.0)
        # Test that it read from 3 channels
        self.assertEqual(seg.analogsignals[0].shape[1], 3)

        self.assertAlmostEqual(seg.t_start.rescale(t_start.units), t_start, delta=5.0)
        self.assertAlmostEqual(seg.t_stop.rescale(t_stop.units), t_stop, delta=5.0)

        r.close()


if __name__ == "__main__":
    unittest.main()
