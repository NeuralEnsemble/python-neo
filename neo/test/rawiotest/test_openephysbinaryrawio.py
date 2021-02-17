import unittest

from neo.rawio.openephysbinaryrawio import OpenEphysBinaryRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestOpenEphysBinaryRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = OpenEphysBinaryRawIO
    entities_to_test = [
        ]

    files_to_download = [
        ]


if __name__ == "__main__":
    unittest.main()
