# -*- coding: utf-8 -*-
"""
Tests of neo.io.neuroexplorerio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import NeuroExplorerIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestNeuroExplorerIO(BaseTestIO, unittest.TestCase, ):
    ioclass = NeuroExplorerIO
    _prefered_signal_group_mode = 'split-all'
    files_to_test = ['File_neuroexplorer_1.nex',
                     'File_neuroexplorer_2.nex',
                     ]
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
