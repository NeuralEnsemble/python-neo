# -*- coding: utf-8 -*-
"""

"""

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.brainvisionrawio import BrainVisionRawIO

from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestBrainVisionRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = BrainVisionRawIO
    entities_to_test = ['File_brainvision_1.vhdr',
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
