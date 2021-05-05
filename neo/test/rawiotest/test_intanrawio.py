import unittest

from neo.rawio.intanrawio import IntanRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestIntanRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = IntanRawIO
    entities_to_download = [
        'intan'
    ]
    entities_to_test = [
        'intan/intan_rhs_test_1.rhs',
        'intan/intan_rhd_test_1.rhd',
    ]


if __name__ == "__main__":
    unittest.main()
