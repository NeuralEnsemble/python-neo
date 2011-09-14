# encoding: utf-8
"""
Tests of io.asciisignalio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import Spike2IO
import numpy

from neo.test.io.common_io_test import BaseTestIO


class TestSpike2IO(BaseTestIO, unittest.TestCase, ):
    ioclass = Spike2IO
    files_to_test = [ 'File_spike2_1.smr',
                            'File_spike2_2.smr',
                            'File_spike2_3.smr',
                            ]
    files_to_download = files_to_test






if __name__ == "__main__":
    unittest.main()
