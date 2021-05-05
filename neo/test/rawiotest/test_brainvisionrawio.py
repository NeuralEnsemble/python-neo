"""

"""

import unittest

from neo.rawio.brainvisionrawio import BrainVisionRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestBrainVisionRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = BrainVisionRawIO
    entities_to_test = [
        'brainvision/File_brainvision_1.vhdr',
        'brainvision/File_brainvision_2.vhdr',
        'brainvision/File_brainvision_3_float32.vhdr',
        'brainvision/File_brainvision_3_int16.vhdr',
        'brainvision/File_brainvision_3_int32.vhdr',
    ]

    entities_to_download = [
        'brainvision'
    ]


if __name__ == "__main__":
    unittest.main()
