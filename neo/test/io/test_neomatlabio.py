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

from neo.test.io.common_io_test import BaseTestIO, download_test_files_if_not_present

class TestNeoMatlabIO(unittest.TestCase, BaseTestIO):
    ioclass = NeoMatlabIO



if __name__ == "__main__":
    unittest.main()
