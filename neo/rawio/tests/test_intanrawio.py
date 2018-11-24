# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.intanrawio import IntanRawIO

from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestIntanRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = IntanRawIO
    files_to_download = [
        'intan_rhs_test_1.rhs',
        'intan_rhd_test_1.rhd',
    ]
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
