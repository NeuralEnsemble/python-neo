"""

"""

import unittest

import quantities as pq

from neo.io import OpenEphysBinaryIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.rawiotest.test_openephysbinaryrawio import TestOpenEphysBinaryRawIO


class TestOpenEphysBinaryIO(BaseTestIO, unittest.TestCase, ):
    ioclass = OpenEphysBinaryIO
    files_to_test = [
        'test_multiple_2020-12-17_17-34-35',
    ]

    files_to_download = TestOpenEphysBinaryRawIO.files_to_download


if __name__ == "__main__":
    unittest.main()
