import unittest

from neo.io import IntanBinaryIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestIntanBinaryIO(BaseTestIO, unittest.TestCase, ):
    ioclass = IntanBinaryIO
    entities_to_download = [
        'intan'
    ]
    entities_to_test = [
        'intan/intan_fpc_test_231117_052630/',
        'intan/intan_fps_test_231117_052500/',
    ]


if __name__ == "__main__":
    unittest.main()
