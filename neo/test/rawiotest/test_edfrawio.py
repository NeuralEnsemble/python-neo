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


if __name__ == "__main__":
    unittest.main()
