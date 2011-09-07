# encoding: utf-8
"""
Tests of io.asciisignalio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import AsciiSpikeTrainIO
import numpy

from neo.test.io.common_io_test import BaseTestIO


class TestAsciiSpikeTrainIO(unittest.TestCase, BaseTestIO):
    ioclass = AsciiSpikeTrainIO
    files_to_test = [ 'File_ascii_spiketrain_1.txt',
                            ]





if __name__ == "__main__":
    unittest.main()
