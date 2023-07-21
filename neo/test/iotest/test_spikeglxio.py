"""
Tests of neo.io.spikeglxio
"""

import unittest

from neo.io import SpikeGLXIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.rawiotest.test_spikeglxrawio import TestSpikeGLXRawIO


class TestSpikeGLXIO(BaseTestIO, unittest.TestCase):
    ioclass = SpikeGLXIO
    entities_to_download = TestSpikeGLXRawIO.entities_to_download
    entities_to_test = TestSpikeGLXRawIO.entities_to_test


if __name__ == "__main__":
    unittest.main()
