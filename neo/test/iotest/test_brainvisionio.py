"""
Tests of neo.io.brainvisionio
"""

import unittest

from neo.io import BrainVisionIO

from neo.test.iotest.common_io_test import BaseTestIO


class TestBrainVisionIO(BaseTestIO, unittest.TestCase, ):
    ioclass = BrainVisionIO
    entities_to_download = [
        'brainvision'
    ]
    entities_to_test = [
        'brainvision/File_brainvision_1.vhdr',
        'brainvision/File_brainvision_2.vhdr',
        'brainvision/File_brainvision_3_float32.vhdr',
        'brainvision/File_brainvision_3_int16.vhdr',
        'brainvision/File_brainvision_3_int32.vhdr',
    ]


if __name__ == "__main__":
    unittest.main()
