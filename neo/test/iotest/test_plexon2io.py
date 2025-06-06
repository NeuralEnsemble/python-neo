"""
Tests of neo.io.plexon2io
"""

import unittest
import os

from neo.io import Plexon2IO
from neo.test.iotest.common_io_test import BaseTestIO

from neo.test.rawiotest.test_plexon2rawio import TestPlexon2RawIO


try:
    from neo.rawio.plexon2rawio.pypl2 import pypl2lib

    HAVE_PYPL2 = True
except (ImportError, TimeoutError):
    HAVE_PYPL2 = False

TEST_PLEXON2 = bool(os.getenv("PLEXON2_TEST"))

@unittest.skipUnless(HAVE_PYPL2 and TEST_PLEXON2, "requires pypl package and all its dependencies")
class TestPlexon2IO(BaseTestIO, unittest.TestCase):
    entities_to_download = TestPlexon2RawIO.entities_to_download
    entities_to_test = TestPlexon2RawIO.entities_to_test
    ioclass = Plexon2IO


if __name__ == "__main__":
    unittest.main()
