# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.rawbinarysignalrawio import RawBinarySignalRawIO
from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestRawBinarySignalRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = RawBinarySignalRawIO
    entities_to_test = ['File_rawbinary_10kHz_2channels_16bit.raw']
    files_to_download = entities_to_test


if __name__ == "__main__":
    unittest.main()
