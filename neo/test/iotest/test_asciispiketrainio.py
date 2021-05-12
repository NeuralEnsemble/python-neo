"""
Tests of neo.io.asciispiketrainio
"""

import unittest

from neo.io import AsciiSpikeTrainIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestAsciiSpikeTrainIO(BaseTestIO, unittest.TestCase, ):
    ioclass = AsciiSpikeTrainIO
    entities_to_download = [
        'asciispiketrain'
    ]
    entities_to_test = [
        'asciispiketrain/File_ascii_spiketrain_1.txt',
    ]


if __name__ == "__main__":
    unittest.main()
