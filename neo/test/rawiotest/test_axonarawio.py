"""
Tests of neo.rawio.axonarawio

Author: Steffen Buergers

"""

import unittest

from neo.rawio.axonarawio import AxonaRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestAxonaRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = AxonaRawIO
    entities_to_download = [
        'axona'
    ]
    entities_to_test = [
        'axona/axona_raw.set',
        'axona/dataset_unit_spikes/20140815-180secs.set',
        'axona/dataset_multi_modal/axona_sample.set'
    ]


if __name__ == "__main__":
    unittest.main()
