"""

"""

import unittest

import quantities as pq

from neo.io import OpenEphysIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.rawiotest.test_openephysrawio import TestOpenEphysRawIO


class TestOpenEphysIO(BaseTestIO, unittest.TestCase, ):
    ioclass = OpenEphysIO
    entities_to_download = TestOpenEphysRawIO.entities_to_download
    entities_to_test = TestOpenEphysRawIO.entities_to_test


if __name__ == "__main__":
    unittest.main()
