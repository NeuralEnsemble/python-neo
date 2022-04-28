import unittest

from neo.rawio.neuroscoperawio import NeuroScopeRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestNeuroScopeRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NeuroScopeRawIO
    entities_to_download = [
        'neuroscope'
    ]
    entities_to_test = [
        'neuroscope/test1/test1',
        'neuroscope/test1/test1.dat',
        'neuroscope/dataset_1/YutaMouse42-151117.eeg'
    ]


if __name__ == "__main__":
    unittest.main()
