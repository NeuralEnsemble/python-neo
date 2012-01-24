# encoding: utf-8
"""
Tests of io.axonio
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import AxonIO
from neo.test.io.common_io_test import BaseTestIO


class TestAxonIO(BaseTestIO, unittest.TestCase):
    files_to_test = ['File_axon_1.abf',
                     'File_axon_2.abf',
                     'File_axon_3.abf',
                     'File_axon_4.abf',
                     'File_axon_5.abf',
                     'File_axon_6.abf',
                     
                     
                        ]
    files_to_download = files_to_test
    ioclass = AxonIO


if __name__ == "__main__":
    unittest.main()
