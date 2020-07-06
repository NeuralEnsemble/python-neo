import unittest

from neo.rawio.neuroscoperawio import NeuroScopeRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestNeuroScopeRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NeuroScopeRawIO
    files_to_download = ['test1/test1.xml',
                         'test1/test1.dat',
                         ]
    entities_to_test = ['test1/test1']


if __name__ == "__main__":
    unittest.main()
