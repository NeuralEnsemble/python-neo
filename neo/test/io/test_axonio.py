# encoding: utf-8
"""
Tests of io.axonio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import AxonIO
import numpy


from neo.test.io.common_io_test import *



files_to_test = [ 'File_axon_1.abf',
                        'File_axon_2.abf',
                        'File_axon_3.abf',
                        'File_axon_4.abf',
                        ]


class TestAxonIO(unittest.TestCase):

    def test_on_files(self):
        localdir = download_test_files_if_not_present(AxonIO,files_to_test )


if __name__ == "__main__":
    unittest.main()
