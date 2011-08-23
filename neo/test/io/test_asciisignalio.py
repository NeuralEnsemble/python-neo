# encoding: utf-8
"""
Tests of io.asciisignalio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import AsciiSignalIO
import numpy

from neo.test.io.common_io_test import *

files_to_test = [ 'File_asciisignal_1.asc',
                        'File_asciisignal_2.txt',
                        'File_asciisignal_3.txt',
                        ]

class TestAsciiSignalIO(unittest.TestCase):
    #~ def test__write_them_read(self):
            #~ test_write_then_read(AsciiSignalIO)

    def test_on_files(self):
        localdir = download_test_files_if_not_present(AsciiSignalIO,files_to_test )






if __name__ == "__main__":
    unittest.main()
