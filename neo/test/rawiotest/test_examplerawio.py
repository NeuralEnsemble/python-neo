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

from neo.rawio.examplerawio import ExampleRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestExampleRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = ExampleRawIO
    # here obliviously there is nothing to download:
    entities_to_download = []

    # here we will test 1 fake file
    # note that for IOs based on directory names you can put the directory
    # name here instead of the filename.
    entities_to_test = ["fake1"]

    def test_buffer_description_api_check(self):
        """An ExampleRawIO reader, whose flag is False, must raise ValueError."""
        reader = ExampleRawIO(filename=self.get_local_path("fake1"))
        reader.parse_header()
        with self.assertRaises(ValueError):
            reader.get_analogsignal_buffer_description(block_index=0, seg_index=0, buffer_id="0")


if __name__ == "__main__":
    unittest.main()
