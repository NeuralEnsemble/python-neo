# -*- coding: utf-8 -*-
"""
Tests of neo.rawio.examplerawio
"""

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.rawio.micromedrawio import MicromedRawIO

from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestMicromedRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = MicromedRawIO
    files_to_test = ['File_micromed_1.TRC']
    files_to_download = files_to_test

if __name__ == "__main__":
    unittest.main()

