# -*- coding: utf-8 -*-
"""
Tests of neo.io.tdtio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division
import unittest
from neo.io import TdtIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.iotest.tools import get_test_file_full_path
import numpy as np


class TestTdtIO(BaseTestIO, unittest.TestCase, ):
    ioclass = TdtIO
    files_to_test = ['aep_05']
    files_to_download = ['aep_05/Block-1/aep_05_Block-1.Tbk',
                         'aep_05/Block-1/aep_05_Block-1.Tdx',
                         'aep_05/Block-1/aep_05_Block-1.tev',
                         'aep_05/Block-1/aep_05_Block-1.tsq',

                         'aep_05/Block-2/aep_05_Block-2.Tbk',
                         'aep_05/Block-2/aep_05_Block-2.Tdx',
                         'aep_05/Block-2/aep_05_Block-2.tev',
                         'aep_05/Block-2/aep_05_Block-2.tsq',

                         # ~ 'aep_05/Block-3/aep_05_Block-3.Tbk',
                         # ~ 'aep_05/Block-3/aep_05_Block-3.Tdx',
                         # ~ 'aep_05/Block-3/aep_05_Block-3.tev',
                         # ~ 'aep_05/Block-3/aep_05_Block-3.tsq',
                         ]

    def test_signal_group_mode(self):
        dirname = get_test_file_full_path(ioclass=TdtIO,
                                          filename='aep_05', directory=self.local_test_dir,
                                          clean=False)

        # TdtIO is a hard case they are 3 groups at rawio level
        # there are 3 groups of signals
        nb_sigs_by_group = [1, 16, 16]

        signal_group_mode = 'group-by-same-units'
        reader = TdtIO(dirname=dirname)
        bl = reader.read_block(signal_group_mode=signal_group_mode)
        for seg in bl.segments:
            assert len(seg.analogsignals) == 3
            i = 0
            for anasig in seg.analogsignals:
                # print(anasig.shape, anasig.sampling_rate)
                assert anasig.shape[1] == nb_sigs_by_group[i]
                i += 1

        signal_group_mode = 'split-all'
        reader = TdtIO(dirname=dirname)
        bl = reader.read_block(signal_group_mode=signal_group_mode)
        for seg in bl.segments:
            assert len(seg.analogsignals) == np.sum(nb_sigs_by_group)
            for anasig in seg.analogsignals:
                # print(anasig.shape, anasig.sampling_rate)
                assert anasig.shape[1] == 1


if __name__ == "__main__":
    unittest.main()
