# -*- coding: utf-8 -*-
"""
Tests of neo.io.asciisignalio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import AsciiSignalIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestAsciiSignalIO(BaseTestIO, unittest.TestCase, ):
    ioclass = AsciiSignalIO
    files_to_download = ['File_asciisignal_1.asc',
                         'File_asciisignal_2.txt',
                         'File_asciisignal_3.txt',
                         ]
    files_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
