# encoding: utf-8
"""
Tests of io.asciisignalio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import AsciiSignalIO
import numpy

from neo.test.io.common_io_test import BaseTestIO


class TestAsciiSignalIO(BaseTestIO, unittest.TestCase, ):
    ioclass = AsciiSignalIO
    files_to_download = [ 'File_asciisignal_1.asc',
                            'File_asciisignal_2.txt',
                            'File_asciisignal_3.txt',
                            ]
    files_to_test = files_to_download






if __name__ == "__main__":
    unittest.main()
