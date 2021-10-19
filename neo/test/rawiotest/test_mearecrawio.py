"""
Tests of neo.rawio.mearecrawio

"""

import unittest

from neo.rawio.mearecrawio import MEArecRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


try:
    import MEArec as mr
    HAVE_MEAREC = True
except ImportError:
    HAVE_MEAREC = False


@unittest.skipUnless(HAVE_MEAREC, "requires MEArec package")
class TestMEArecRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = MEArecRawIO
    entities_to_download = [
        'mearec'
    ]
    entities_to_test = [
        'mearec/mearec_test_10s.h5'
    ]


if __name__ == "__main__":
    unittest.main()
