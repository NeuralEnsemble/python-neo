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
        'spikeglx/Noise4Sam_g0',
        'spikeglx/TEST_20210920_0_g0'
    ]

    def test_with_location(self):
        rawio = SpikeGLXRawIO(self.get_local_path('spikeglx/Noise4Sam_g0'), load_channel_location=True)
        rawio.parse_header()
        # one of the stream have channel location
        have_location = []
        for sig_anotations in rawio.raw_annotations['blocks'][0]['segments'][0]['signals']:
            have_location.append('channel_location_0' in sig_anotations['__array_annotations__'])
        assert any(have_location)


if __name__ == "__main__":
    unittest.main()
