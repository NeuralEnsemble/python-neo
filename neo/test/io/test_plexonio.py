# encoding: utf-8
"""
Tests of io.plexon
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import PlexonIO
import numpy

from neo.test.io.common_io_test import download_test_files_if_not_present


files_to_test = [   'File_plexon_1.plx',
                            'File_plexon_2.plx',
                            'File_plexon_3.plx',
                        ]
                        

class TestPlexonIO(unittest.TestCase):
    def test_on_files(self):
        localdir = download_test_files_if_not_present(PlexonIO,files_to_test )
        




if __name__ == "__main__":
    unittest.main()
