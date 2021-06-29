"""

"""

import unittest

from neo.rawio.micromedrawio import MicromedRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestMicromedRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = MicromedRawIO
    entities_to_download = [
        'micromed'
    ]
    entities_to_test = [
        'micromed/File_micromed_1.TRC'
    ]


if __name__ == "__main__":
    unittest.main()
