import unittest

import numpy as np

from neo.rawio.neuralynxrawio import NeuralynxRawIO
from neo.rawio.neuralynxrawio.NlxHeader import NlxHeader
from neo.rawio.neuralynxrawio.NcsBlocks import (NcsBlocks, NcsBlocksFactory)
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

import logging

logging.getLogger().setLevel(logging.INFO)


class TestNeuralynxRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NeuralynxRawIO
    entities_to_test = [
        'BML/original_data',
        'BML_unfilledsplit/original_data',
        'Cheetah_v1.1.0/original_data',
        'Cheetah_v4.0.2/original_data',
        'Cheetah_v5.5.1/original_data',
        'Cheetah_v5.6.3/original_data',
        'Cheetah_v5.7.4/original_data',
        'Cheetah_v6.3.2/incomplete_blocks']
    files_to_download = [
        'BML/original_data/CSC1_trunc.Ncs',
        'BML/plain_data/CSC1_trunc.txt',
        'BML/README.txt',
        'BML_unfilledsplit/original_data/unfilledSplitRecords.Ncs',
        'BML_unfilledsplit/plain_data/unfilledSplitRecords.txt',
        'BML_unfilledsplit/README.txt',
        'Cheetah_v1.1.0/original_data/CSC67_trunc.Ncs',
        'Cheetah_v1.1.0/README.txt',
        'Cheetah_v1.1.0/plain_data/CSC67_trunc.txt',
        'Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs',
        'Cheetah_v4.0.2/plain_data/CSC14_trunc.txt',
        'Cheetah_v4.0.2/README.txt',
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

    def test_scan_ncs_files(self):

        # Test BML style of Ncs files, similar to PRE4 but with fractional frequency
        # in the header and fractional microsPerSamp, which is then rounded as appropriate
        # in each record.
        rawio = NeuralynxRawIO(self.get_filename_path('BML/original_data'))
        rawio.parse_header()
        # test values here from direct inspection of .ncs files
        self.assertEqual(rawio._nb_segment, 1)
        self.assertListEqual(rawio._ncs_seg_timestamp_limits.timestamp_limits, [(0, 192000)])
        self.assertEqual(rawio._ncs_seg_timestamp_limits.length[0], 4608)
        self.assertEqual(rawio._ncs_seg_timestamp_limits.t_start[0], 0)
        self.assertEqual(rawio._ncs_seg_timestamp_limits.t_stop[0], 0.192)
        self.assertEqual(len(rawio._ncs_memmaps), 1)

        # Test Cheetah 4.0.2, which is PRE4 type with frequency in header and
        # no microsPerSamp. Number of microseconds per sample in file is inverse of
        # sampling frequency in header trucated to microseconds.
        rawio = NeuralynxRawIO(self.get_filename_path('Cheetah_v4.0.2/original_data'))
        rawio.parse_header()
        # test values here from direct inspection of .ncs files
        self.assertEqual(rawio._nb_segment, 1)
        self.assertListEqual(rawio._ncs_seg_timestamp_limits.timestamp_limits,
                             [(266982936, 267162136)])
        self.assertEqual(rawio._ncs_seg_timestamp_limits.length[0], 5120)
        self.assertEqual(rawio._ncs_seg_timestamp_limits.t_start[0], 266.982936)
        self.assertEqual(rawio._ncs_seg_timestamp_limits.t_stop[0], 267.162136)
        self.assertEqual(len(rawio._ncs_memmaps), 1)

        # Test Cheetah 5.5.1, which is DigitalLynxSX and has two blocks of records
        # with a fairly large gap.
        rawio = NeuralynxRawIO(self.get_filename_path('Cheetah_v5.5.1/original_data'))
        rawio.parse_header()
        # test values here from direct inspection of .ncs files
        self.assertEqual(rawio._nb_segment, 2)
        self.assertListEqual(rawio._ncs_seg_timestamp_limits.timestamp_limits,
                             [(26122557633, 26162525633), (26366360633, 26379704633)])
        self.assertListEqual(rawio._ncs_seg_timestamp_limits.length, [1278976, 427008])
        self.assertListEqual(rawio._ncs_seg_timestamp_limits.t_stop, [26162.525633, 26379.704633])
        self.assertListEqual(rawio._ncs_seg_timestamp_limits.t_start, [26122.557633, 26366.360633])
        self.assertEqual(len(rawio._ncs_memmaps), 2)  # check only that there are 2 memmaps

        # Test Cheetah 6.3.2, the incomplete_blocks test. This is a DigitalLynxSX with
        # three blocks of records. Gaps are on the order of 60 microseconds or so.
        rawio = NeuralynxRawIO(self.get_filename_path('Cheetah_v6.3.2/incomplete_blocks'))
        rawio.parse_header()
        # test values here from direct inspection of .ncs file, except for 3rd block
        # t_stop, which is extended due to events past the last block of ncs records.
        self.assertEqual(rawio._nb_segment, 3)
        self.assertListEqual(rawio._ncs_seg_timestamp_limits.timestamp_limits,
                             [(8408806811, 8427831990), (8427832053, 8487768498),
                              (8487768561, 8515816549)])
        self.assertListEqual(rawio._ncs_seg_timestamp_limits.length, [608806, 1917967, 897536])
        self.assertListEqual(rawio._ncs_seg_timestamp_limits.t_stop, [8427.831990, 8487.768498,
                                                                      8515.816549])
        self.assertListEqual(rawio._ncs_seg_timestamp_limits.t_start, [8408.806811, 8427.832053,
                                                                       8487.768561])
        self.assertEqual(len(rawio._ncs_memmaps), 3)  # check only that there are 3 memmaps


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
            hdr = NlxHeader.build_for_file(filename)
            self.assertEqual(hdr.type_of_recording(), typeTest[1])


class TestNcsBlocksFactory(TestNeuralynxRawIO, unittest.TestCase):
    """
    Test building NcsBlocks for files of different revisions.
    """

    def test_ncsblocks_partial(self):
        filename = self.get_filename_path('Cheetah_v6.3.2/incomplete_blocks/CSC1_reduced.ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        self.assertEqual(data0.shape[0], 6690)
        self.assertEqual(data0['timestamp'][6689], 8515800549)  # timestamp of last record

        hdr = NlxHeader.build_for_file(filename)
        nb = NcsBlocksFactory.buildForNcsFile(data0, hdr)
        self.assertEqual(nb.sampFreqUsed, 32000.012813673042)
        self.assertEqual(nb.microsPerSampUsed, 31.249987486652431)
        self.assertListEqual([blk.startBlock for blk in nb.blocks], [0, 1190, 4937])
        self.assertListEqual([blk.endBlock for blk in nb.blocks], [1189, 4936, 6689])

    def testBuildGivenActualFrequency(self):

        # Test early files where the frequency listed in the header is
        # floor(1e6/(actual number of microseconds between samples)
        filename = self.get_filename_path('Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        ncsBlocks = NcsBlocks()
        ncsBlocks.sampFreqUsed = 1 / (35e-6)
        ncsBlocks.microsPerSampUsed = 35
        ncsBlocks = NcsBlocksFactory._buildGivenActualFrequency(data0, ncsBlocks.sampFreqUsed,
                                                                27789)
        self.assertEqual(len(ncsBlocks.blocks), 1)
        self.assertEqual(ncsBlocks.blocks[0].startBlock, 0)
        self.assertEqual(ncsBlocks.blocks[0].endBlock, 9)

    def testBuildUsingHeaderAndScanning(self):

        # Test early files where the frequency listed in the header is
        # floor(1e6/(actual number of microseconds between samples)
        filename = self.get_filename_path('Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs')
        hdr = NlxHeader.build_for_file(filename)
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        nb = NcsBlocksFactory.buildForNcsFile(data0, hdr)

        self.assertEqual(nb.sampFreqUsed, 1 / 35e-6)
        self.assertEqual(nb.microsPerSampUsed, 35)
        self.assertEqual(len(nb.blocks), 1)
        self.assertEqual(nb.blocks[0].startBlock, 0)
        self.assertEqual(nb.blocks[0].endBlock, 9)

        # test Cheetah 5.5.1, which is DigitalLynxSX and has two blocks of records
        # with a fairly large gap
        filename = self.get_filename_path('Cheetah_v5.5.1/original_data/Tet3a.ncs')
        hdr = NlxHeader.build_for_file(filename)
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        nb = NcsBlocksFactory.buildForNcsFile(data0, hdr)
        self.assertEqual(nb.sampFreqUsed, 32000)
        self.assertEqual(nb.microsPerSampUsed, 31.25)
        self.assertEqual(len(nb.blocks), 2)
        self.assertListEqual([blk.startBlock for blk in nb.blocks], [0, 2498])
        self.assertListEqual([blk.endBlock for blk in nb.blocks], [2497, 3331])

    def testBlockStartAndEndTimes(self):
        # digitallynxsx version to exercise the _parseForMaxGap function with multiple blocks
        filename = self.get_filename_path('Cheetah_v6.3.2/incomplete_blocks/CSC1_reduced.ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr = NlxHeader.build_for_file(filename)
        nb = NcsBlocksFactory.buildForNcsFile(data0, hdr)
        self.assertListEqual([blk.startTime for blk in nb.blocks], [8408806811, 8427832053,
                                                                    8487768561])
        self.assertListEqual([blk.endTime for blk in nb.blocks], [8427831990, 8487768498,
                                                                  8515816549])

        # digitallynxsx with single block of records to exercise path in _buildForMaxGap
        filename = self.get_filename_path('Cheetah_v1.1.0/original_data/CSC67_trunc.Ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr = NlxHeader.build_for_file(filename)
        nb = NcsBlocksFactory.buildForNcsFile(data0, hdr)
        self.assertEqual(len(nb.blocks), 1)
        self.assertEqual(nb.blocks[0].startTime, 253293161778)
        self.assertEqual(nb.blocks[0].endTime, 253293349278)

        # PRE4 version with single block of records to exercise path in _buildGivenActualFrequency
        filename = self.get_filename_path('Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr = NlxHeader.build_for_file(filename)
        nb = NcsBlocksFactory.buildForNcsFile(data0, hdr)
        self.assertEqual(len(nb.blocks), 1)
        self.assertEqual(nb.blocks[0].startTime, 266982936)
        self.assertEqual(nb.blocks[0].endTime, 267162136)

        # BML style with two blocks of records and one partially filled record to exercise
        # _parseGivenActualFrequency
        filename = self.get_filename_path(
            'BML_unfilledsplit/original_data/unfilledSplitRecords.Ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr = NlxHeader.build_for_file(filename)
        nb = NcsBlocksFactory.buildForNcsFile(data0, hdr)
        self.assertEqual(len(nb.blocks), 2)
        self.assertListEqual([blk.startTime for blk in nb.blocks], [1837623129, 6132625241])
        self.assertListEqual([blk.endTime for blk in nb.blocks], [1837651009, 6132642649])

    def testBlockVerify(self):
        # check that file verifies against itself for single block
        filename = self.get_filename_path('Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr0 = NlxHeader.build_for_file(filename)
        nb0 = NcsBlocksFactory.buildForNcsFile(data0, hdr0)

        self.assertTrue(NcsBlocksFactory._verifyBlockStructure(data0, nb0))

        # check that fails against file with two blocks
        filename = self.get_filename_path(
            'BML_unfilledsplit/original_data/unfilledSplitRecords.Ncs')
        data1 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr1 = NlxHeader.build_for_file(filename)
        nb1 = NcsBlocksFactory.buildForNcsFile(data1, hdr1)

        self.assertFalse(NcsBlocksFactory._verifyBlockStructure(data1, nb0))

        # check that two blocks verify against self
        self.assertTrue(NcsBlocksFactory._verifyBlockStructure(data1, nb1))


if __name__ == "__main__":
    unittest.main()
