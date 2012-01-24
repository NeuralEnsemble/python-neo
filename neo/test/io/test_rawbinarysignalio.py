# encoding: utf-8
"""
Tests of io.asciisignalio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import RawBinarySignalIO
import numpy


from neo.test.io.common_io_test import BaseTestIO



class TestRawBinarySignalIO(BaseTestIO, unittest.TestCase, ):
    ioclass = RawBinarySignalIO
    files_to_test = [ 'File_rawbinary_10kHz_2channels_16bit.raw',
                            ]
    files_to_download = files_to_test




if __name__ == "__main__":
    unittest.main()
