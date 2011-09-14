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


from neo.test.io.common_io_test import BaseTestIO



class TestRawBinarySignalIO(BaseTestIO , unittest.TestCase, ):
    ioclass =  WinWcpIO
    files_to_test = [   'File_winwcp_1.wcp',
                            ]
    files_to_download = files_to_test
    



if __name__ == "__main__":
    unittest.main()
