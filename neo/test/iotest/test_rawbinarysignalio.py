# -*- coding: utf-8 -*-
"""
Tests of io.rawbinarysignal
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import unittest

from neo.io import RawBinarySignalIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestRawBinarySignalIO(BaseTestIO, unittest.TestCase, ):
    ioclass = RawBinarySignalIO
    files_to_test = ['File_rawbinary_10kHz_2channels_16bit.raw']
    files_to_download = files_to_test

    # Skip  this test because there are not mechanisim
    # in neo.tests.iotests to set parameters for the ioclass
    # in a generic way. In that case nb_channel must be given
    # to pass the the set
    def test_write_then_read(self):
        pass


if __name__ == "__main__":
    unittest.main()
