# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.neuralynxrawio import NeuralynxRawIO
from neo.rawio.tests.common_rawio_test import BaseTestRawIO

import logging

logging.getLogger().setLevel(logging.INFO)


class TestNeuralynxRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NeuralynxRawIO
    entities_to_test = [
        'Cheetah_v5.5.1/original_data',
        'Cheetah_v5.6.3/original_data',
        'Cheetah_v5.7.4/original_data',
    ]
    files_to_download = [
        'Cheetah_v5.5.1/original_data/CheetahLogFile.txt',
        'Cheetah_v5.5.1/original_data/CheetahLostADRecords.txt',
        'Cheetah_v5.5.1/original_data/Events.nev',
        'Cheetah_v5.5.1/original_data/STet3a.nse',
        'Cheetah_v5.5.1/original_data/STet3b.nse',
        'Cheetah_v5.5.1/original_data/Tet3a.ncs',
        'Cheetah_v5.5.1/original_data/Tet3b.ncs',
        'Cheetah_v5.5.1/plain_data/STet3a.txt',
        'Cheetah_v5.5.1/plain_data/STet3b.txt',
        'Cheetah_v5.5.1/plain_data/Tet3a.txt',
        'Cheetah_v5.5.1/plain_data/Tet3b.txt',
        'Cheetah_v5.5.1/plain_data/Events.txt',
        'Cheetah_v5.5.1/README.txt',
        'Cheetah_v5.6.3/original_data/CheetahLogFile.txt',
        'Cheetah_v5.6.3/original_data/CheetahLostADRecords.txt',
        'Cheetah_v5.6.3/original_data/Events.nev',
        'Cheetah_v5.6.3/original_data/CSC1.ncs',
        'Cheetah_v5.6.3/original_data/CSC2.ncs',
        'Cheetah_v5.6.3/original_data/TT1.ntt',
        'Cheetah_v5.6.3/original_data/TT2.ntt',
        'Cheetah_v5.6.3/original_data/VT1.nvt',
        'Cheetah_v5.6.3/plain_data/Events.txt',
        'Cheetah_v5.6.3/plain_data/CSC1.txt',
        'Cheetah_v5.6.3/plain_data/CSC2.txt',
        'Cheetah_v5.6.3/plain_data/TT1.txt',
        'Cheetah_v5.6.3/plain_data/TT2.txt',
        'Cheetah_v5.7.4/original_data/CSC1.ncs',
        'Cheetah_v5.7.4/original_data/CSC2.ncs',
        'Cheetah_v5.7.4/original_data/CSC3.ncs',
        'Cheetah_v5.7.4/original_data/CSC4.ncs',
        'Cheetah_v5.7.4/original_data/CSC5.ncs',
        'Cheetah_v5.7.4/original_data/Events.nev',
        'Cheetah_v5.7.4/plain_data/CSC1.txt',
        'Cheetah_v5.7.4/plain_data/CSC2.txt',
        'Cheetah_v5.7.4/plain_data/CSC3.txt',
        'Cheetah_v5.7.4/plain_data/CSC4.txt',
        'Cheetah_v5.7.4/plain_data/CSC5.txt',
        'Cheetah_v5.7.4/plain_data/Events.txt',
        'Cheetah_v5.7.4/README.txt']


if __name__ == "__main__":
    unittest.main()
