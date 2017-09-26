# -*- coding: utf-8 -*-
"""
Tests of neo.io.plexonio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import sys

import unittest

from neo.io import PlexonIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestPlexonIO(BaseTestIO, unittest.TestCase, ):
    ioclass = PlexonIO
    files_to_test = [
                    'File_plexon_1.plx',
                    'File_plexon_2.plx',
                    'File_plexon_3.plx',
                     ]
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
