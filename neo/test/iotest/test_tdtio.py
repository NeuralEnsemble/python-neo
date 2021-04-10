"""
Tests of neo.io.tdtio
"""

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
