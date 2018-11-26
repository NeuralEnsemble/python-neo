# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.openephysrawio import OpenEphysRawIO
from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestOpenEphysRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = OpenEphysRawIO
    entities_to_test = ['OpenEphys_SampleData_1',
        # 'OpenEphys_SampleData_2_(multiple_starts)',  # This not implemented this raise error
        # 'OpenEphys_SampleData_3',
                        ]

    files_to_download = [
        # One segment
        'OpenEphys_SampleData_1/101_CH0.continuous',
        'OpenEphys_SampleData_1/101_CH1.continuous',
        'OpenEphys_SampleData_1/all_channels.events',
        'OpenEphys_SampleData_1/Continuous_Data.openephys',
        'OpenEphys_SampleData_1/messages.events',
        'OpenEphys_SampleData_1/settings.xml',
        'OpenEphys_SampleData_1/STp106.0n0.spikes',

        # Multi segment with multi file
        # NOT implemented for now in the IO
        # Raise Error
        'OpenEphys_SampleData_2_(multiple_starts)/101_CH0_2.continuous',
        'OpenEphys_SampleData_2_(multiple_starts)/101_CH1_2.continuous',
        'OpenEphys_SampleData_2_(multiple_starts)/all_channels_2.events',
        'OpenEphys_SampleData_2_(multiple_starts)/Continuous_Data_2.openephys',
        'OpenEphys_SampleData_2_(multiple_starts)/messages_2.events',
        'OpenEphys_SampleData_2_(multiple_starts)/settings_2.xml',
        'OpenEphys_SampleData_2_(multiple_starts)/STp106.0n0_2.spikes',
        'OpenEphys_SampleData_2_(multiple_starts)/101_CH0.continuous',
        'OpenEphys_SampleData_2_(multiple_starts)/101_CH1.continuous',
        'OpenEphys_SampleData_2_(multiple_starts)/all_channels.events',
        'OpenEphys_SampleData_2_(multiple_starts)/Continuous_Data.openephys',
        'OpenEphys_SampleData_2_(multiple_starts)/messages.events',
        'OpenEphys_SampleData_2_(multiple_starts)/settings.xml',
        'OpenEphys_SampleData_2_(multiple_starts)/STp106.0n0.spikes',

        # Multi segment with corrupted file (CH32) : implemenetd
        'OpenEphys_SampleData_3/100_CH1_2.continuous',
        'OpenEphys_SampleData_3/100_CH2_2.continuous',
        'OpenEphys_SampleData_3/100_CH32_2.continuous',
        'OpenEphys_SampleData_3/100_CH32.continuous',
        'OpenEphys_SampleData_3/all_channels_2.events',
        'OpenEphys_SampleData_3/Continuous_Data_2.openephys',
        'OpenEphys_SampleData_3/messages_2.events',
        'OpenEphys_SampleData_3/settings_2.xml',
        'OpenEphys_SampleData_3/100_CH1.continuous',
        'OpenEphys_SampleData_3/100_CH2.continuous',
        'OpenEphys_SampleData_3/100_CH3_2.continuous',
        'OpenEphys_SampleData_3/100_CH3.continuous',
        'OpenEphys_SampleData_3/all_channels.events',
        'OpenEphys_SampleData_3/Continuous_Data.openephys',
        'OpenEphys_SampleData_3/messages.events',
        'OpenEphys_SampleData_3/settings.xml',
    ]

    def test_raise_error_if_discontinuous_files(self):
        # the case of discontinuous signals is NOT cover by the IO for the moment
        # It must raise an error
        reader = OpenEphysRawIO(dirname=self.get_filename_path(
            'OpenEphys_SampleData_2_(multiple_starts)'))
        with self.assertRaises(Exception):
            reader.parse_header()

    def test_raise_error_if_strange_timestamps(self):
        # In this dataset CH32 have strange timestamps
        reader = OpenEphysRawIO(dirname=self.get_filename_path('OpenEphys_SampleData_3'))
        with self.assertRaises(Exception):
            reader.parse_header()


if __name__ == "__main__":
    unittest.main()
