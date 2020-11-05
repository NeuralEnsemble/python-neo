"""
Tests of neo.io.spikeglxio
"""

import unittest

from neo.io import SpikeGLXIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestSpikeGLXIO(BaseTestIO, unittest.TestCase):
    files_to_test = [
    ]
    files_to_download = files_to_test
    ioclass = SpikeGLXIO


if __name__ == "__main__":
    unittest.main()
