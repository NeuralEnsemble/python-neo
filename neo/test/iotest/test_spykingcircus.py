"""
Tests of neo.rawio.spykingcircusio
"""

import unittest

from neo.io.spykingcircusio import SpykingCircusIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestSpykingCircusIO(BaseTestIO, unittest.TestCase):
    files_to_download = [
        'spykingcircus_example0/recording.params',
        'spykingcircus_example0/recording/recording.result.hdf5',
        'spykingcircus_example0/recording/recording.result-merged.hdf5',
    ]
    entities_to_test = ['spykingcircus_example0']
    ioclass = SpykingCircusIO


if __name__ == "__main__":
    unittest.main()
