# -*- coding: utf-8 -*-
"""

"""

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.rawio.spike2rawio import Spike2RawIO

from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestSpike2RawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = Spike2RawIO
    files_to_test = [
                    'File_spike2_1.smr',
                     'File_spike2_2.smr',
                     'File_spike2_3.smr',
                     '130322-1LY.smr', # this is for bug 182
                     ]
    files_to_download = files_to_test

if __name__ == "__main__":
    unittest.main()

