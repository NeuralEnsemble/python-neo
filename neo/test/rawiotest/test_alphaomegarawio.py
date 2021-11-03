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

from neo.rawio.alphaomegarawio import AlphaOmegaRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestAlphaOmegaRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = AlphaOmegaRawIO

    entities_to_download = [
        "alphaomega",
    ]

    entities_to_test = [
        "alphaomega/",
    ]


if __name__ == "__main__":
    unittest.main()
