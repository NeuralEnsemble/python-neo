import unittest

from neo.rawio.tdtrawio import TdtRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestTdtRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = TdtRawIO
    entities_to_test = ['aep_05']

    files_to_download = [
        'aep_05/Block-1/aep_05_Block-1.Tbk',
        'aep_05/Block-1/aep_05_Block-1.Tdx',
        'aep_05/Block-1/aep_05_Block-1.tev',
        'aep_05/Block-1/aep_05_Block-1.tsq',

        'aep_05/Block-2/aep_05_Block-2.Tbk',
        'aep_05/Block-2/aep_05_Block-2.Tdx',
        'aep_05/Block-2/aep_05_Block-2.tev',
        'aep_05/Block-2/aep_05_Block-2.tsq',

    ]


if __name__ == "__main__":
    unittest.main()
