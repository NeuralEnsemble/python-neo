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

from neo.test.io.common_io_test import BaseTestIO, download_test_files_if_not_present

files_to_test = [ 'File_ascii_spiketrain_1.txt',
                        ]

class TestAsciiSpikeTrainIO(unittest.TestCase, BaseTestIO):
    ioclass = AsciiSpikeTrainIO
    
    def test_on_files(self):
        localdir = download_test_files_if_not_present(AsciiSpikeTrainIO, files_to_test)






if __name__ == "__main__":
    unittest.main()
