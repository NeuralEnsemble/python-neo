"""
Tests of neo.rawio.spikeglxrawio
"""

import unittest

from neo.rawio.tridesclousrawio import TridesclousRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestTrisdesclousRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = TridesclousRawIO
    files_to_download = [
        'Noise4Sam_g0/Noise4Sam_g0_t0.nidq.bin',
        'Noise4Sam_g0/Noise4Sam_g0_t0.nidq.meta',
        'Noise4Sam_g0/Noise4Sam_g0_imec0/Noise4Sam_g0_t0.imec0.ap.bin',
        'Noise4Sam_g0/Noise4Sam_g0_imec0/Noise4Sam_g0_t0.imec0.ap.meta',
        'Noise4Sam_g0/Noise4Sam_g0_imec0/Noise4Sam_g0_t0.imec0.lf.bin',
        'Noise4Sam_g0/Noise4Sam_g0_imec0/Noise4Sam_g0_t0.imec0.lf.meta'
    ]
    entities_to_test = ['Noise4Sam_g0']


if __name__ == "__main__":
    unittest.main()