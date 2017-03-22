# -*- coding: utf-8 -*-
"""
Tests of neo.io.spike2io
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import Spike2IO
from neo.test.iotest.common_io_test import BaseTestIO


class TestSpike2IO(BaseTestIO, unittest.TestCase, ):
    ioclass = Spike2IO
    files_to_test = ['File_spike2_1.smr',
                     'File_spike2_2.smr',
                     'File_spike2_3.smr',
                     '130322-1LY.smr', # this is for bug 182
                     ]
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
