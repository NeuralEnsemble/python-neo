"""
Tests of neo.io.spikeglxio
"""

import unittest

from neo.io import SpikeGLXIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestSpikeGLXIO(BaseTestIO, unittest.TestCase):
    files_to_test = ['Noise4Sam_g0']
    files_to_download = [
        'Noise4Sam_g0/Noise4Sam_g0_t0.nidq.bin',
        'Noise4Sam_g0/Noise4Sam_g0_t0.nidq.meta',
        'Noise4Sam_g0/Noise4Sam_g0_imec0/Noise4Sam_g0_t0.imec0.ap.bin',
        'Noise4Sam_g0/Noise4Sam_g0_imec0/Noise4Sam_g0_t0.imec0.ap.meta',
        'Noise4Sam_g0/Noise4Sam_g0_imec0/Noise4Sam_g0_t0.imec0.lf.bin',
        'Noise4Sam_g0/Noise4Sam_g0_imec0/Noise4Sam_g0_t0.imec0.lf.meta'
    ]
    ioclass = SpikeGLXIO


if __name__ == "__main__":
    unittest.main()
