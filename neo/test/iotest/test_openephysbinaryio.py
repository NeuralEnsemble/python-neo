"""

"""

import unittest

import quantities as pq

from neo.io import OpenEphysBinaryIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.rawiotest.test_openephysbinaryrawio import TestOpenEphysBinaryRawIO


class TestOpenEphysBinaryIO(BaseTestIO, unittest.TestCase):
    ioclass = OpenEphysBinaryIO
    entities_to_download = TestOpenEphysBinaryRawIO.entities_to_download
    entities_to_test = TestOpenEphysBinaryRawIO.entities_to_test


if __name__ == "__main__":
    unittest.main()
