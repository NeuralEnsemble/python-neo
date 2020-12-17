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

import unittest

from neo.rawio.mearecrawio import MEArecRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestMEArecRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = MEArecRawIO
    # here obsvisously there is nothing to download:
    files_to_download = ['mearec_test_10s.h5']
    # here we will test 2 fake files
    # not that IO base on dirname you can put the dirname here.
    entities_to_test = ['mearec_test_10s.h5']


if __name__ == "__main__":
    unittest.main()
