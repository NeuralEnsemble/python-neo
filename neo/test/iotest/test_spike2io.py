# -*- coding: utf-8 -*-
"""
Tests of neo.io.spike2io
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import unittest

import quantities as pq

from neo.io import Spike2IO
from neo.test.iotest.common_io_test import BaseTestIO


class TestSpike2IO(BaseTestIO, unittest.TestCase, ):
    ioclass = Spike2IO
    files_to_test = [
        'File_spike2_1.smr',
        'File_spike2_2.smr',
        'File_spike2_3.smr',
        '130322-1LY.smr',  # this is for bug 182
        'multi_sampling.smr',  # this is for bug 466
    ]
    files_to_download = files_to_test

    def test_multi_sampling(self):
        """
        Some file can have several sampling_rate.
        This one contain 3 differents signals sampling rate
        """
        filename = self.get_filename_path('multi_sampling.smr')
        reader = Spike2IO(filename=filename)
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


if __name__ == "__main__":
    unittest.main()
