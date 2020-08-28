import unittest

from neo.rawio.rawbinarysignalrawio import RawBinarySignalRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestRawBinarySignalRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = RawBinarySignalRawIO
    entities_to_test = ['File_rawbinary_10kHz_2channels_16bit.raw']
    files_to_download = entities_to_test


if __name__ == "__main__":
    unittest.main()
