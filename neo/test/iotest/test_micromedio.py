"""
Tests of neo.io.neomatlabio
"""

import unittest

from neo.io import MicromedIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestMicromedIO(BaseTestIO, unittest.TestCase, ):
    ioclass = MicromedIO
    entities_to_download = [
        'micromed'
    ]
    entities_to_test = [
        'micromed/File_micromed_1.TRC'
    ]


if __name__ == "__main__":
    unittest.main()
