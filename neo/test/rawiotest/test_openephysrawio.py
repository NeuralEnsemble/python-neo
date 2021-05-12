import unittest

from neo.rawio.openephysrawio import OpenEphysRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestOpenEphysRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = OpenEphysRawIO
    entities_to_download = [
        'openephys'
    ]
    entities_to_test = [
        'openephys/OpenEphys_SampleData_1',
        # 'OpenEphys_SampleData_2_(multiple_starts)',  # This not implemented this raise error
        # 'OpenEphys_SampleData_3',
    ]

    def test_raise_error_if_discontinuous_files(self):
        # the case of discontinuous signals is NOT cover by the IO for the moment
        # It must raise an error
        reader = OpenEphysRawIO(dirname=self.get_local_path(
            'openephys/OpenEphys_SampleData_2_(multiple_starts)'))
        with self.assertRaises(Exception):
            reader.parse_header()

    def test_raise_error_if_strange_timestamps(self):
        # In this dataset CH32 have strange timestamps
        reader = OpenEphysRawIO(dirname=self.get_local_path('openephys/OpenEphys_SampleData_3'))
        with self.assertRaises(Exception):
            reader.parse_header()

    def test_channel_order(self):
        reader = OpenEphysRawIO(dirname=self.get_local_path(
            'openephys/OpenEphys_SampleData_1'))
        reader.parse_header()
        reader.header['signal_channels']['name'][0].startswith('CH')


if __name__ == "__main__":
    unittest.main()
