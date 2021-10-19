"""
Tests of neo.rawio.spykingcircusrawio
"""

import unittest

from neo.rawio.spykingcircusrawio import SpykingCircusRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestSpykingCircusRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = SpykingCircusRawIO
    files_to_download = [
        'spykingcircus/spykingcircus_example0/recording.params',
        'spykingcircus/spykingcircus_example0/recording/recording.result.hdf5',
        'spykingcircus/spykingcircus_example0/recording/recording.result-merged.hdf5',
    ]
    entities_to_test = ['spykingcircus']


if __name__ == "__main__":
    unittest.main()
