import unittest

from neo.io import MaxwellIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestMaxwellIO(BaseTestIO, unittest.TestCase, ):
    ioclass = MaxwellIO
    files_to_test = []
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
