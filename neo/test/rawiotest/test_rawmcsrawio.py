import unittest

from neo.rawio.rawmcsrawio import RawMCSRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestRawMCSRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = RawMCSRawIO
    entities_to_test = ['raw_mcs_with_header_1.raw']
    files_to_download = entities_to_test


if __name__ == "__main__":
    unittest.main()
