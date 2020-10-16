import unittest

import numpy as np

from neo.rawio.neuralynxrawio import NeuralynxRawIO
from neo.rawio.neuralynxrawio import NlxHeader
from neo.rawio.neuralynxrawio import NcsBlocksFactory
from neo.rawio.neuralynxrawio import NcsBlocks
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

import logging

logging.getLogger().setLevel(logging.INFO)


class TestNeuralynxRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NeuralynxRawIO
    entities_to_test = [
        # 'Cheetah_v4.0.2/original_data',
        'Cheetah_v5.5.1/original_data',
        'Cheetah_v5.6.3/original_data',
        'Cheetah_v5.7.4/original_data',
        'Cheetah_v6.3.2/incomplete_blocks']
    files_to_download = [
        'Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs',
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
        'Cheetah_v5.7.4/README.txt',
        'Cheetah_v6.3.2/incomplete_blocks/CSC1_reduced.ncs',
        'Cheetah_v6.3.2/incomplete_blocks/Events.nev',
        'Cheetah_v6.3.2/incomplete_blocks/README.txt']


class TestNcsRecordingType(TestNeuralynxRawIO, unittest.TestCase):
    """
    Test of decoding of NlxHeader for type of recording.
    """

    ncsTypeTestFiles = [
        ('Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs', 'PRE4'),
        ('Cheetah_v5.5.1/original_data/STet3a.nse', 'DIGITALLYNXSX'),
        ('Cheetah_v5.5.1/original_data/Tet3a.ncs', 'DIGITALLYNXSX'),
        ('Cheetah_v5.6.3/original_data/CSC1.ncs', 'DIGITALLYNXSX'),
        ('Cheetah_v5.6.3/original_data/TT1.ntt', 'DIGITALLYNXSX'),
        ('Cheetah_v5.7.4/original_data/CSC1.ncs', 'DIGITALLYNXSX'),
        ('Cheetah_v6.3.2/incomplete_blocks/CSC1_reduced.ncs', 'DIGITALLYNXSX')]

    def test_recording_types(self):

        for typeTest in self.ncsTypeTestFiles:

            filename = self.get_filename_path(typeTest[0])
            hdr = NlxHeader.buildForFile(filename)
            self.assertEqual(hdr.typeOfRecording(), typeTest[1])

class TestNcsBlocksFactory(TestNeuralynxRawIO, unittest.TestCase):
    """
    Test building NcsBlocks for files of different revisions.
    """

    def test_ncsblocks_partial(self):
        filename = self.get_filename_path('Cheetah_v6.3.2/incomplete_blocks/CSC1_reduced.ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                           offset=NlxHeader.HEADER_SIZE)
        self.assertEqual(data0.shape[0],6690)
        self.assertEqual(data0['timestamp'][6689],8515800549) # timestamp of last record

    def testBuildGivenActualFrequency(self):

        # Test early files where the frequency listed in the header is
        # floor(1e6/(actual number of microseconds between samples)
        filename = self.get_filename_path('Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                           offset=NlxHeader.HEADER_SIZE)
        ncsBlocks = NcsBlocks()
        ncsBlocks.sampFreqUsed = 1/(35e-6)
        ncsBlocks.microsPerSampUsed = 35
        ncsBlocks = NcsBlocksFactory._buildGivenActualFrequency(data0, ncsBlocks.sampFreqUsed, 27789)
        self.assertEqual(len(ncsBlocks.startBlocks), 1)
        self.assertEqual(ncsBlocks.startBlocks[0], 0)
        self.assertEqual(len(ncsBlocks.endBlocks), 1)
        self.assertEqual(ncsBlocks.endBlocks[0], 9)

if __name__ == "__main__":
    unittest.main()
