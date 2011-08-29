# encoding: utf-8
"""
Tests of io.winwcp
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import WinWcpIO
import numpy


from neo.test.io.common_io_test import BaseTestIO, download_test_files_if_not_present

files_to_test = [   'File_winwcp_1.wcp',
                        ]


class TestRawBinarySignalIO(unittest.TestCase, BaseTestIO):
    ioclass =  WinWcpIO
    
    def test_on_files(self):
        localdir = download_test_files_if_not_present(WinWcpIO, files_to_test)




if __name__ == "__main__":
    unittest.main()
