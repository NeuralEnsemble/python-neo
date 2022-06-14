"""
Tests of neo.io.spikeglxio
"""

import unittest

from neo.io import SpikeGLXIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestSpikeGLXIO(BaseTestIO, unittest.TestCase):
    ioclass = SpikeGLXIO
    entities_to_download = [
        'spikeglx'
    ]
    entities_to_test = [
        'spikeglx/Noise4Sam_g0',
        'spikeglx/TEST_20210920_0_g0'
    ]
    


if __name__ == "__main__":
    unittest.main()
