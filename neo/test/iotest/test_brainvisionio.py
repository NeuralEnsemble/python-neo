# -*- coding: utf-8 -*-
"""
Tests of neo.io.brainvisionio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import BrainVisionIO

from neo.test.iotest.common_io_test import BaseTestIO


class TestBrainVisionIO(BaseTestIO, unittest.TestCase, ):
    ioclass = BrainVisionIO
    files_to_test = ['File_brainvision_1.vhdr',
                     'File_brainvision_2.vhdr',
                     'File_brainvision_3_float32.vhdr',
                     'File_brainvision_3_int16.vhdr',
                     'File_brainvision_3_int32.vhdr',
                     ]
    files_to_download = ['File_brainvision_1.eeg',
                         'File_brainvision_1.vhdr',
                         'File_brainvision_1.vmrk',
                         'File_brainvision_2.eeg',
                         'File_brainvision_2.vhdr',
                         'File_brainvision_2.vmrk',
                         'File_brainvision_3_float32.eeg',
                         'File_brainvision_3_float32.vhdr',
                         'File_brainvision_3_float32.vmrk',
                         'File_brainvision_3_int16.eeg',
                         'File_brainvision_3_int16.vhdr',
                         'File_brainvision_3_int16.vmrk',
                         'File_brainvision_3_int32.eeg',
                         'File_brainvision_3_int32.vhdr',
                         'File_brainvision_3_int32.vmrk',
                         ]


if __name__ == "__main__":
    unittest.main()
