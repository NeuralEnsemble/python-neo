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


@unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
class TestNeuroExplorerIO(BaseTestIO, unittest.TestCase, ):
    ioclass = NeuroExplorerIO
    files_to_test = ['File_neuroexplorer_1.nex',
                     'File_neuroexplorer_2.nex',
                     ]
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
