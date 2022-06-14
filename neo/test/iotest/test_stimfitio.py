# -*- coding: utf-8 -*-
"""
Tests of neo.io.stimfitio
"""

import sys

import unittest

from neo.io import StimfitIO
from neo.test.iotest.common_io_test import BaseTestIO

try:
    import stfio
except Exception:
    HAS_STFIO = False
else:
    HAS_STFIO = True


@unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
@unittest.skipUnless(HAS_STFIO, "requires stfio")
class TestStimfitIO(BaseTestIO, unittest.TestCase):
    ioclass = StimfitIO
    entities_to_download = [
        'stimfit'
    ]
    entities_to_test = [
        'stimfit/File_stimfit_1.h5',
        'stimfit/File_stimfit_2.h5',
        'stimfit/File_stimfit_3.h5',
        'stimfit/File_stimfit_4.h5',
        'stimfit/File_stimfit_5.h5',
        'stimfit/File_stimfit_6.h5',
    ]


if __name__ == "__main__":
    unittest.main()
