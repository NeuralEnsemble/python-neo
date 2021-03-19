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
        'axona_raw.set', 
        'axona_raw.bin'
    ]
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
