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
        seg = reader.read_segment(signal_group_mode = 'group-by-same-units')
        
        # 3 group_id
        assert len(seg.analogsignals) == 3
        
        # 1 channel for 1kHz
        assert seg.analogsignals[0].shape == (135812, 1)
        assert seg.analogsignals[0].sampling_rate == 1000*pq.Hz

        # 4  channel for 1kHz
        assert seg.analogsignals[1].shape == (264846, 4)
        assert seg.analogsignals[1].sampling_rate == 2000*pq.Hz

        # 2 channel for 10kHz
        assert seg.analogsignals[2].shape == (1146180, 2)
        assert seg.analogsignals[2].sampling_rate == 10000*pq.Hz



if __name__ == "__main__":
    unittest.main()
