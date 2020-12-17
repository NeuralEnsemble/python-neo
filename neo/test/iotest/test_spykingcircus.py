"""

"""

import unittest

from neo.io.spykingcircusio import SpykingCircusIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestSpykingCircusIO(BaseTestIO, unittest.TestCase):
    files_to_test = []
    files_to_download = [
    ]
    ioclass = SpykingCircusIO


if __name__ == "__main__":
    unittest.main()
