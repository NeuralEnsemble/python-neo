"""
Tests of neo.rawio.edfrawio
"""

import unittest

from neo.rawio.edfrawio import EDFRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestExampleRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = EDFRawIO
    entities_to_download = ['edf']
    entities_to_test = [
        'edf/edf+C.edf',
    ]

    def test_context_handler(self):
        filename = self.get_local_path('edf/edf+C.edf')
        with EDFRawIO(filename) as io:
            io.parse_header()

        # Check that file was closed properly and can be opened again
        with open(filename) as f:
            pass

    def test_close(self):
        filename = self.get_local_path('edf/edf+C.edf')

        # Open file and close it again
        io1 = EDFRawIO(filename)
        io1.parse_header()
        io1.close()

        # Check that file was closed properly and can be opened again
        io2 = EDFRawIO(filename)
        io2.parse_header()
        io2.close()


if __name__ == "__main__":
    unittest.main()
