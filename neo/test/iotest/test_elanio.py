# -*- coding: utf-8 -*-
"""
Tests of neo.io.elanio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import ElanIO
from neo.test.iotest.common_io_test import BaseTestIO


@unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
class TestElanIO(BaseTestIO, unittest.TestCase, ):
    ioclass = ElanIO
    files_to_test = ['File_elan_1.eeg']
    files_to_download = ['File_elan_1.eeg',
                         'File_elan_1.eeg.ent',
                         'File_elan_1.eeg.pos',
                         ]


if __name__ == "__main__":
    unittest.main()
