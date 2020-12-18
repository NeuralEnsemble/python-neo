"""

"""

import unittest

from neo.io import TridesclousIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.rawiotest.test_tridesclousrawio import TestTrisdesclousRawIO


class TestTridesclousIO(BaseTestIO, unittest.TestCase):
    files_to_test = TestTrisdesclousRawIO.entities_to_test
    files_to_download = TestTrisdesclousRawIO.files_to_download
    ioclass = TridesclousIO


if __name__ == "__main__":
    unittest.main()
