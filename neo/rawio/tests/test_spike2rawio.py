# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.spike2rawio import Spike2RawIO

from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestSpike2RawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = Spike2RawIO
    files_to_download = [
        'File_spike2_1.smr',
        'File_spike2_2.smr',
        'File_spike2_3.smr',
        '130322-1LY.smr',  # this is for bug 182
        'multi_sampling.smr',  # this is for bug 466
    ]
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
