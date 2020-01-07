"""
Tests of neo.io.asciispiketrainio
"""

import unittest

from neo.io import AsciiSpikeTrainIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestAsciiSpikeTrainIO(BaseTestIO, unittest.TestCase, ):
    ioclass = AsciiSpikeTrainIO
    files_to_download = ['File_ascii_spiketrain_1.txt']
    files_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
