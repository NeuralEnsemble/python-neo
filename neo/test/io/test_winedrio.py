# encoding: utf-8
"""
Tests of io.wineedrio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import WinEdrIO
import numpy


from neo.test.io.common_io_test import BaseTestIO, download_test_files_if_not_present

files_to_test = [   'File_WinEDR_1.EDR',
                            'File_WinEDR_2.EDR',
                            'File_WinEDR_3.EDR',
                        ]


class TestRawBinarySignalIO(unittest.TestCase, BaseTestIO):
    ioclass = WinEdrIO

    def test_on_files(self):
        localdir = download_test_files_if_not_present(WinEdrIO, files_to_test)




if __name__ == "__main__":
    unittest.main()
