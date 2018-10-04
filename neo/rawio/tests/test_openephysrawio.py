# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.openephysrawio import OpenEphysRawIO
from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestOpenEphysRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = OpenEphysRawIO
    entities_to_test = ['OpenEphys_SampleData_1']

    files_to_download = [
        'OpenEphys_SampleData_1/101_CH0.continuous',
        'OpenEphys_SampleData_1/101_CH1.continuous',
        'OpenEphys_SampleData_1/all_channels.events',
        'OpenEphys_SampleData_1/Continuous_Data.openephys',
        'OpenEphys_SampleData_1/messages.events',
        'OpenEphys_SampleData_1/settings.xml',
        'OpenEphys_SampleData_1/STp106.0n0.spikes',
    ]


if __name__ == "__main__":
    unittest.main()


            