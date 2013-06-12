# encoding: utf-8
"""
Tests of io.plexon
"""
from __future__ import absolute_import, division
import sys
try:
    import unittest2 as unittest
except ImportError:
    import unittest

from ...io import PlexonIO
import numpy

from .common_io_test import BaseTestIO




@unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
class TestPlexonIO(BaseTestIO, unittest.TestCase, ):
    ioclass = PlexonIO
    files_to_test = [   'File_plexon_1.plx',
                                'File_plexon_2.plx',
                                'File_plexon_3.plx',
                            ]
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
