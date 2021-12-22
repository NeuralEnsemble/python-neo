import unittest

from neo.rawio.tdtrawio import TdtRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestTdtRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = TdtRawIO
    entities_to_download = [
        'tdt'
    ]
    entities_to_test = [
        'tdt/aep_05',
        'tdt/aep_05/Block-1/aep_05_Block-1.Tdx'
    ]


if __name__ == "__main__":
    unittest.main()
