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

from neo.rawio.examplerawio import ExampleRawIO

from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestExampleRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = ExampleRawIO
    files_to_test = ['fake1',
                     'fake2',
                     ]
    files_to_download = []


if __name__ == "__main__":
    unittest.main()

