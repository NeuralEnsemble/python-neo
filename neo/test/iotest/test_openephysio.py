# -*- coding: utf-8 -*-
"""

"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import unittest

import quantities as pq

from neo.io import OpenEphysIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.rawio.tests.test_openephysrawio import TestOpenEphysRawIO


class TestOpenEphysIO(BaseTestIO, unittest.TestCase, ):
    ioclass = OpenEphysIO
    files_to_test = ['OpenEphys_SampleData_1',
        # 'OpenEphys_SampleData_2_(multiple_starts)',  # This not implemented this raise error
        # 'OpenEphys_SampleData_3',
                     ]

    files_to_download = TestOpenEphysRawIO.files_to_download


if __name__ == "__main__":
    unittest.main()
