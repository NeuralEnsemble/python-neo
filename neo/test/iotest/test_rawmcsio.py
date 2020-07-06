import unittest

from neo.io import RawMCSIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestRawMcsIO(BaseTestIO, unittest.TestCase, ):
    ioclass = RawMCSIO
    files_to_test = ['raw_mcs_with_header_1.raw']
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
