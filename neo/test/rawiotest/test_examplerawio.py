# -*- coding: utf-8 -*-
"""
Tests of neo.rawio.examplerawio

Note for dev:
if you write a new RawIO class your need to put some file
to be tested at g-node portal, Ask neuralensemble list for that.
The file need to be small.

Then you have to copy/paste/renamed the TestExampleRawIO
class and a full test will be done to test if the new coded IO
is compliant with the RawIO API.

If you have problems, do not hesitate to ask help github (prefered)
of neuralensemble list.

Note that same mechanism is used a neo.io API so files are tested
several time with neo.rawio (numpy buffer) and neo.io (neo object tree).
See neo.test.iotest.*


Author: Samuel Garcia

"""

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.examplerawio import ExampleRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestExampleRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = ExampleRawIO
    # here obsvisously there is nothing to download:
    files_to_download = []
    # here we will test 2 fake files
    # not that IO base on dirname you can put the dirname here.
    entities_to_test = ['fake1',
                        'fake2',
                        ]


if __name__ == "__main__":
    unittest.main()
