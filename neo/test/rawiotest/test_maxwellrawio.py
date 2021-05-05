import unittest

from neo.rawio.maxwellrawio import MaxwellRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestMaxwellRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = MaxwellRawIO
    entities_to_download = [
        'maxwell'
    ]
    entities_to_test = files_to_test = [
        'maxwell/MaxOne_data/Record/000011/data.raw.h5',
        'maxwell/MaxTwo_data/Network/000028/data.raw.h5'
    ]


if __name__ == "__main__":
    unittest.main()
