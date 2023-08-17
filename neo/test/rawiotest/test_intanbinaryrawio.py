import unittest

from neo.rawio.intanbinaryrawio import IntanBinaryRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestIntanBinaryRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = IntanBinaryRawIO
    entities_to_download = [
        'intan'
    ]
    entities_to_test = [
        'intan/simulated_intan_230816_095232/',
        'intan/simulated_intan_230816_182040/',
    ]


if __name__ == "__main__":
    unittest.main()