# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.axographrawio import AxoGraphRawIO

from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestAxoGraphRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = AxoGraphRawIO
    files_to_download = [
        'File_axograph.axgd',
    ]
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
