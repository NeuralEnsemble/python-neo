"""
Tests of neo.rawio.spikeglxrawio
"""

import unittest

from neo.rawio.spikeglxrawio import SpikeGLXRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestSpikeGLXRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = SpikeGLXRawIO
    entities_to_download = [
        'spikeglx'
    ]
    entities_to_test = [
        'spikeglx/Noise4Sam_g0/Noise4Sam_g0_imec0'
    ]


if __name__ == "__main__":
    unittest.main()
