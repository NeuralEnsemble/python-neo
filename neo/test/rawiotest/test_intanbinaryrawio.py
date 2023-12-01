import unittest

from neo.rawio.intanbinaryrawio import IntanBinaryRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestIntanBinaryRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = IntanBinaryRawIO
    entities_to_download = [
        'intan'
    ]
    entities_to_test = [
        'intan/intan_fpc_test_231117_052630/',
        'intan/intan_fps_test_231117_052500/',
    ]


if __name__ == "__main__":
    unittest.main()
