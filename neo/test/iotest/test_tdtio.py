"""
Tests of neo.io.tdtio
"""

import unittest
from neo.io import TdtIO
from neo.test.iotest.common_io_test import BaseTestIO
import numpy as np


class TestTdtIO(BaseTestIO, unittest.TestCase, ):
    ioclass = TdtIO
    entities_to_download = [
        'tdt'
    ]
    entities_to_test = [
        # test structure directory with multiple blocks
        'tdt/aep_05',
        # test single block
        'tdt/dataset_0_single_block/512ch_reconly_all-181123_B24_rest.Tdx',
        'tdt/dataset_1_single_block/ECTest-220207-135355_ECTest_B1.Tdx',
        'tdt/aep_05/Block-1/aep_05_Block-1.Tdx'
    ]

    def test_signal_group_mode(self):
        dirname = self.get_local_path('tdt/aep_05')

        # In this TDT dataset there are 3 signal streams
        nb_sigs_by_stream = [16, 1, 16]

        reader = TdtIO(dirname=dirname)
        bl = reader.read_block()
        for seg in bl.segments:
            assert len(seg.analogsignals) == 3
            i = 0
            for anasig in seg.analogsignals:
                # print(anasig.shape, anasig.sampling_rate, nb_sigs_by_stream[i])
                assert anasig.shape[1] == nb_sigs_by_stream[i]
                i += 1


if __name__ == "__main__":
    unittest.main()
