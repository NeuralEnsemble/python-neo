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

from neo.test.io.common_io_test import BaseTestIO

class TestNeoMatlabIO(unittest.TestCase, BaseTestIO):
    ioclass = NeoMatlabIO
    files_to_test = [ ]



if __name__ == "__main__":
    unittest.main()
