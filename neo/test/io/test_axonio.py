# encoding: utf-8
"""
Tests of io.axonio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import AxonIO
import numpy


from neo.test.io.common_io_test import BaseTestIO



class TestAxonIO(unittest.TestCase, BaseTestIO):
    files_to_test = [ 'File_axon_1.abf',
                            'File_axon_2.abf',
                            'File_axon_3.abf',
                            'File_axon_4.abf',
                            ]
    ioclass = AxonIO


if __name__ == "__main__":
    unittest.main()
