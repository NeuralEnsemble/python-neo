"""
Tests of neo.io.wineedrio
"""

import unittest

from neo.io import WinEdrIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestWinedrIO(BaseTestIO, unittest.TestCase, ):
    ioclass = WinEdrIO
    files_to_test = ['File_WinEDR_1.EDR',
                     'File_WinEDR_2.EDR',
                     'File_WinEDR_3.EDR',
                     ]
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
