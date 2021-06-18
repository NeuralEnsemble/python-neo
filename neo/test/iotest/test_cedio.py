import unittest

from neo.io import CedIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestCedIO(BaseTestIO, unittest.TestCase, ):
    ioclass = CedIO
    entities_to_test = [
        'spike2/m365_1sec.smrx',
        'spike2/File_spike2_1.smr',
        'spike2/Two-mice-bigfile-test000.smr'
    ]
    entities_to_download = [
        'spike2'
    ]


if __name__ == "__main__":
    unittest.main()
