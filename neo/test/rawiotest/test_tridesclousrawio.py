"""
Tests of neo.rawio.spikeglxrawio
"""

import unittest

from neo.rawio.tridesclousrawio import TridesclousRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestTrisdesclousRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = TridesclousRawIO
    files_to_download = [
        'tdc_example0/info.json',
        'tdc_example0/probe.prb',
        'tdc_example0/channel_group_0/segment_0/arrays.json',
        'tdc_example0/channel_group_0/segment_0/spikes.raw',
        'tdc_example0/channel_group_0/segment_0/processed_signals.raw',
        'tdc_example0/channel_group_0/catalogues/initial/arrays.json',
        'tdc_example0/channel_group_0/catalogues/initial/catalogue.pickle',
        'tdc_example0/channel_group_0/catalogues/initial/clusters.raw',
    ]
    entities_to_test = ['tdc_example0']


if __name__ == "__main__":
    unittest.main()
