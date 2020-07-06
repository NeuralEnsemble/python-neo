import unittest

from neo.rawio.intanrawio import IntanRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestIntanRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = IntanRawIO
    files_to_download = [
        'intan_rhs_test_1.rhs',
        'intan_rhd_test_1.rhd',
    ]
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
