"""
Tests of neo.rawio.spikeglxrawio
"""

import unittest

from neo.rawio.spikeglxrawio import SpikeGLXRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestSpikeGLXRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = SpikeGLXRawIO
    files_to_download = [
    ]
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
