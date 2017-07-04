# -*- coding: utf-8 -*-
"""
Tests of neo.rawio.examplerawio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.rawio.examplerawio import ExampleRawIO

from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestExampleIO(BaseTestRawIO, unittest.TestCase, ):
    ioclass = ExampleRawIO
    files_to_test = ['fake1',
                     'fake2',
                     ]
    files_to_download = []