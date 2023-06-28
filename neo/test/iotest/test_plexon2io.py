"""
Tests of neo.io.plexon2io
"""

import unittest

from neo.io import Plexon2IO
from neo.test.iotest.common_io_test import BaseTestIO

from neo.test.rawiotest.test_plexon2rawio import TestPlexon2RawIO


class TestPlexon2IO(BaseTestIO, unittest.TestCase):
    entities_to_download = TestPlexon2RawIO.entities_to_download
    entities_to_test = TestPlexon2RawIO.entities_to_test
    ioclass = Plexon2IO


if __name__ == "__main__":
    unittest.main()
