"""

"""

import unittest

from neo.io import TridesclousIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestTridesclousIO(BaseTestIO, unittest.TestCase):
    files_to_test = []
    files_to_download = [
    ]
    ioclass = TridesclousIO


if __name__ == "__main__":
    unittest.main()
