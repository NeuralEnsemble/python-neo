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

from neo.test.io.common_io_test import test_write_them_read

class TestNeoMatlabIO(unittest.TestCase):
    def test__write_them_read(self):
            test_write_them_read(NeoMatlabIO)





if __name__ == "__main__":
    unittest.main()
