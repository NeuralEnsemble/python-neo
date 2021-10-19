"""
Tests of neo.io.neuroexplorerio
"""

import unittest

from neo.io import NeuroExplorerIO
from neo.test.iotest.common_io_test import BaseTestIO

class TestNeuroExplorerIO(BaseTestIO, unittest.TestCase, ):
    ioclass = NeuroExplorerIO
    entities_to_download = [
        'neuroexplorer'
    ]
    entities_to_download_test = [
        'neuroexplorer/File_neuroexplorer_1.nex',
        'neuroexplorer/File_neuroexplorer_2.nex',
     ]

    def test_signal_group_mode(self):
        filename = self.get_local_path('neuroexplorer/File_neuroexplorer_1.nex')

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
