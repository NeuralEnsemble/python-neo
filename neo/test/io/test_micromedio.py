# encoding: utf-8
"""
Tests of io.asciisignalio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import MicromedIO
import numpy

from neo.test.io.common_io_test import BaseTestIO



class TestMicromedIO(unittest.TestCase, BaseTestIO):
    ioclass = MicromedIO
    files_to_test = [ 'File_micromed_1.TRC',
                            ]
    


if __name__ == "__main__":
    unittest.main()
