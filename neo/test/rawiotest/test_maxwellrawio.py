import unittest

from neo.rawio.maxwellrawio import MaxwellRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestMaxwellRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = MaxwellRawIO
    entities_to_test = []
    files_to_download = entities_to_test


if __name__ == "__main__":
    unittest.main()
