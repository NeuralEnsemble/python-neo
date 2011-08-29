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


from neo.test.io.common_io_test import BaseTestIO, download_test_files_if_not_present

files_to_test = [ 'File_rawbinary_10kHz_2channels_16bit.raw',
                        ]


class TestRawBinarySignalIO(unittest.TestCase, BaseTestIO):
    ioclass = RawBinarySignalIO

    def test_on_files(self):
        localdir = download_test_files_if_not_present(RawBinarySignalIO, files_to_test)




if __name__ == "__main__":
    unittest.main()
