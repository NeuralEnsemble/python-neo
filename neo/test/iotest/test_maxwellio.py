import unittest

from neo.io import MaxwellIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestMaxwellIO(BaseTestIO, unittest.TestCase, ):
    ioclass = MaxwellIO
    entities_to_download = [
        'maxwell'
    ]
    entities_to_test = files_to_test = [
        'maxwell/MaxOne_data/Record/000011/data.raw.h5',
        'maxwell/MaxTwo_data/Network/000028/data.raw.h5'
    ]


if __name__ == "__main__":
    unittest.main()
