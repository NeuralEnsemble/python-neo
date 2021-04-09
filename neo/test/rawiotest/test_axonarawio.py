"""
Tests of neo.rawio.axonarawio

Author: Steffen Buergers

"""

import unittest

from neo.rawio.axonarawio import AxonaRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestAxonaRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = AxonaRawIO
    files_to_download = [
        'axona_raw.bin',
        'axona_raw.set',
    ]
    entities_to_test = [
        'axona_raw.bin'
    ]


if __name__ == "__main__":
    unittest.main()
