"""
Tests of neo.io.wineedrio
"""

import unittest

from neo.io import WinEdrIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestWinedrIO(BaseTestIO, unittest.TestCase, ):
    ioclass = WinEdrIO
    entities_to_download = [
        'winedr'
    ]
    entities_to_test = [
        'winedr/File_WinEDR_1.EDR',
        'winedr/File_WinEDR_2.EDR',
        'winedr/File_WinEDR_3.EDR',
    ]


if __name__ == "__main__":
    unittest.main()
