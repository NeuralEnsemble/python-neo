# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.rawmcsrawio import RawMCSRawIO
from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestRawMCSRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = RawMCSRawIO
    entities_to_test = ['raw_mcs_with_header_1.raw']
    files_to_download = entities_to_test


if __name__ == "__main__":
    unittest.main()
