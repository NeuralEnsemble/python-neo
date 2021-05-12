import unittest

from neo.io import RawMCSIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestRawMcsIO(BaseTestIO, unittest.TestCase, ):
    ioclass = RawMCSIO
    entities_to_download = [
        'rawmcs'
    ]
    entities_to_test = [
        'rawmcs/raw_mcs_with_header_1.raw'
    ]


if __name__ == "__main__":
    unittest.main()
