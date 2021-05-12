import unittest

from neo.rawio.rawmcsrawio import RawMCSRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestRawMCSRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = RawMCSRawIO
    entities_to_download = [
        'rawmcs'
    ]
    entities_to_test = [
        'rawmcs/raw_mcs_with_header_1.raw'
    ]


if __name__ == "__main__":
    unittest.main()
