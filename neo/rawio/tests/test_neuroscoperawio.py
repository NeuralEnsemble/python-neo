# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.neuroscoperawio import NeuroScopeRawIO
from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestNeuroScopeRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NeuroScopeRawIO
    files_to_download = ['test1/test1.xml',
                         'test1/test1.dat',
                         ]
    entities_to_test = ['test1/test1']


if __name__ == "__main__":
    unittest.main()
