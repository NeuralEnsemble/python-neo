"""
Tests of neo.io.spike2io
"""

import unittest

import quantities as pq

from neo.io import Spike2IO
from neo.test.iotest.common_io_test import BaseTestIO


class TestSpike2IO(BaseTestIO, unittest.TestCase, ):
    ioclass = Spike2IO
    entities_to_download = [
        'spike2'
    ]
    entities_to_test = [
        'spike2/File_spike2_1.smr',
        'spike2/File_spike2_2.smr',
        'spike2/File_spike2_3.smr',
        'spike2/130322-1LY.smr',  # this is for bug 182
        'spike2/multi_sampling.smr',  # this is for bug 466
        'spike2/Two-mice-bigfile-test000.smr',  # SONv9 file
    ]

    def test_multi_sampling_no_grouping(self):
        """
        Some file can have several sampling_rate.
        This one contain 3 differents signals sampling rate
        """
        filename = self.get_local_path('spike2/multi_sampling.smr')
        reader = Spike2IO(filename=filename, try_signal_grouping=False)
        bl = reader.read_block(signal_group_mode='group-by-same-units')
        assert len(bl.segments) == 10
        seg = bl.segments[0]

        # 7 group_id one per channel
        assert len(seg.analogsignals) == 7

        # 1 channel for 1kHz
        assert seg.analogsignals[0].shape == (14296, 1)
        assert seg.analogsignals[0].sampling_rate == 1000 * pq.Hz

        # 4  channel for 2kHz
        for c in range(1, 5):
            assert seg.analogsignals[c].shape == (28632, 1)
            assert seg.analogsignals[c].sampling_rate == 2000 * pq.Hz

        # 2 channel for 10kHz
        for c in range(5, 7):
            assert seg.analogsignals[c].shape == (114618, 1)
            assert seg.analogsignals[c].sampling_rate == 10000 * pq.Hz

    def test_multi_sampling_no_grouping(self):
        """
        Some files can contain multiple sampling rates.
        This file contains three signals with different sampling rates.
        """
        filename = self.get_local_path('spike2/multi_sampling.smr')
        reader = Spike2IO(filename=filename, try_signal_grouping=True)
        bl = reader.read_block(signal_group_mode='group-by-same-units')
        assert len(bl.segments) == 10
        seg = bl.segments[0]

        # 3 groups
        assert len(seg.analogsignals) == 3

        # 1 channel for 1kHz
        assert seg.analogsignals[0].shape == (14296, 1)
        assert seg.analogsignals[0].sampling_rate == 1000 * pq.Hz

        # 4  channel for 2kHz
        assert seg.analogsignals[1].shape == (28632, 4)
        assert seg.analogsignals[1].sampling_rate == 2000 * pq.Hz

        # 2 channel for 10kHz
        assert seg.analogsignals[2].shape == (114618, 2)
        assert seg.analogsignals[2].sampling_rate == 10000 * pq.Hz


if __name__ == "__main__":
    unittest.main()
