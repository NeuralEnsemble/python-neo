import unittest

from neo.io import IntanBinaryIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestIntanBinaryIO(BaseTestIO, unittest.TestCase, ):
    ioclass = IntanBinaryIO
    entities_to_download = [
        'intan'
    ]
    entities_to_test = [
        'intan/simulated_intan_230816_095232/',
        'intan/simulated_intan_230816_182040/',
    ]


if __name__ == "__main__":
    unittest.main()