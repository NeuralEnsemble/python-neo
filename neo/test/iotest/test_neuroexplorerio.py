# -*- coding: utf-8 -*-
"""
Tests of neo.io.neuroexplorerio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import sys

import unittest

from neo.io import NeuroExplorerIO
from neo.test.iotest.common_io_test import BaseTestIO

from neo.test.iotest.tools import get_test_file_full_path


class TestNeuroExplorerIO(BaseTestIO, unittest.TestCase, ):
    ioclass = NeuroExplorerIO
    _prefered_signal_group_mode = 'split-all'
    files_to_test = ['File_neuroexplorer_1.nex',
                     'File_neuroexplorer_2.nex',
                     ]
    files_to_download = files_to_test

    def test_signal_group_mode(self):
        filename = get_test_file_full_path(ioclass=NeuroExplorerIO,
                                           filename='File_neuroexplorer_1.nex',
                                           directory=self.local_test_dir,
                                           clean=False)

        # test that 2 signals are rendered with 2 sampling_rate
        for signal_group_mode in ('group-by-same-units', 'split-all'):
            reader = NeuroExplorerIO(filename=filename)
            bl = reader.read_block(signal_group_mode=signal_group_mode)
            seg = bl.segments[0]
            assert len(seg.analogsignals) == 2
            anasig0 = seg.analogsignals[0]
            anasig1 = seg.analogsignals[1]
            assert anasig0.sampling_rate != anasig1.sampling_rate
            assert anasig0.shape != anasig1.shape
            # ~ for anasig in seg.analogsignals:
            # ~ print(anasig.shape, anasig.sampling_rate)


if __name__ == "__main__":
    unittest.main()
