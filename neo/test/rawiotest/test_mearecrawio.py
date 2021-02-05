"""
Tests of neo.rawio.mearecrawio

"""

import unittest

from neo.rawio.mearecrawio import MEArecRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestMEArecRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = MEArecRawIO
    files_to_download = ['mearec_test_10s.h5']
    entities_to_test = ['mearec_test_10s.h5']


if __name__ == "__main__":
    unittest.main()
