# encoding: utf-8
"""
Tests of io.asciisignalio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import NeoMatlabIO
import numpy

from neo.test.io.common_io_test import test_write_then_read, test_read_then_write

class TestNeoMatlabIO(unittest.TestCase):
    def test__write_then_read(self):
            test_write_then_read(NeoMatlabIO)
    
    def test__write_them_read(self):
            test_read_then_write(NeoMatlabIO)
        
    





if __name__ == "__main__":
    unittest.main()
