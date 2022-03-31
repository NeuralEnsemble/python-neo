import unittest

import numpy as np

from neo.rawio.neuralynxrawio.neuralynxrawio import NeuralynxRawIO
from neo.rawio.neuralynxrawio.nlxheader import NlxHeader
from neo.rawio.neuralynxrawio.ncssections import (NcsSection, NcsSections,
                                                  NcsSectionsFactory)
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

import logging

logging.getLogger().setLevel(logging.INFO)


class TestNeuralynxRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NeuralynxRawIO
    entities_to_download = [
        'neuralynx'
    ]
    entities_to_test = [
        'neuralynx/BML/original_data',
        'neuralynx/BML_unfilledsplit/original_data',
        'neuralynx/Cheetah_v1.1.0/original_data',
        'neuralynx/Cheetah_v4.0.2/original_data',
        'neuralynx/Cheetah_v5.4.0/original_data',
        'neuralynx/Cheetah_v5.5.1/original_data',
        'neuralynx/Cheetah_v5.6.3/original_data',
        'neuralynx/Cheetah_v5.7.4/original_data',
        'neuralynx/Cheetah_v6.3.2/incomplete_blocks']

    def test_scan_ncs_files(self):

        # Test BML style of Ncs files, similar to PRE4 but with fractional frequency
        # in the header and fractional microsPerSamp, which is then rounded as appropriate
        # in each record.
        rawio = NeuralynxRawIO(self.get_local_path('neuralynx/BML/original_data'))
        rawio.parse_header()
        # test values here from direct inspection of .ncs files
        self.assertEqual(rawio._nb_segment, 1)
        self.assertListEqual(rawio._timestamp_limits, [(0, 192000)])
        self.assertEqual(rawio._sigs_length[0][('unknown', '1')], 4608)
        self.assertListEqual(rawio._signal_limits, [(0, 0.192)])
        self.assertEqual(len(rawio._sigs_memmaps), 1)

        # Test Cheetah 4.0.2, which is PRE4 type with frequency in header and
        # no microsPerSamp. Number of microseconds per sample in file is inverse of
        # sampling frequency in header trucated to microseconds.
        rawio = NeuralynxRawIO(self.get_local_path('neuralynx/Cheetah_v4.0.2/original_data'))
        rawio.parse_header()
        # test values here from direct inspection of .ncs files
        self.assertEqual(rawio._nb_segment, 1)
        self.assertEqual(rawio.signal_streams_count(), 1)
        self.assertListEqual(rawio._timestamp_limits, [(266982936, 267162136)])
        self.assertEqual(rawio._sigs_length[0][('unknown', '13')], 5120)
        self.assertListEqual(rawio._signal_limits, [(266.982936, 267.162136)])
        self.assertEqual(len(rawio._sigs_memmaps), 1)

        # Test Cheetah 5.5.1, which is DigitalLynxSX and has two blocks of records
        # with a fairly large gap.
        rawio = NeuralynxRawIO(self.get_local_path('neuralynx/Cheetah_v5.5.1/original_data'))
        rawio.parse_header()
        # test values here from direct inspection of .ncs files
        self.assertEqual(rawio._nb_segment, 2)
        self.assertEqual(rawio.signal_streams_count(), 1)
        self.assertListEqual(rawio._timestamp_limits, [(26122557633, 26162525633),
                                                       (26366360633, 26379704633)])
        self.assertDictEqual(rawio._sigs_length[0],
                             {('Tet3a', '8'): 1278976, ('Tet3b', '9'): 1278976})
        self.assertDictEqual(rawio._sigs_length[1],
                             {('Tet3a', '8'): 427008, ('Tet3b', '9'): 427008})
        self.assertListEqual(rawio._signal_limits, [(26122.557633, 26162.525633),
                                                    (26366.360633, 26379.704633)])
        self.assertEqual(len(rawio._sigs_memmaps), 2)  # check only that there are 2 memmaps

        # Test Cheetah 6.3.2, the incomplete_blocks test. This is a DigitalLynxSX with
        # three blocks of records. Gaps are on the order of 60 microseconds or so.
        rawio = NeuralynxRawIO(self.get_local_path('neuralynx/Cheetah_v6.3.2/incomplete_blocks'))
        rawio.parse_header()
        # test values here from direct inspection of .ncs file, except for 3rd block
        # t_stop, which is extended due to events past the last block of ncs records.
        self.assertEqual(rawio._nb_segment, 3)
        self.assertEqual(rawio.signal_streams_count(), 1)
        self.assertListEqual(rawio._timestamp_limits, [(8408806811, 8427831990),
                                                       (8427832053, 8487768498),
                                                       (8487768561, 8515816549)])
        self.assertDictEqual(rawio._sigs_length[0], {('CSC1', '48'): 608806})
        self.assertDictEqual(rawio._sigs_length[1], {('CSC1', '48'): 1917967})
        self.assertDictEqual(rawio._sigs_length[2], {('CSC1', '48'): 897536})
        self.assertListEqual(rawio._signal_limits, [(8408.806811, 8427.831990),
                                                  (8427.832053, 8487.768498),
                                                  (8487.768561, 8515.816549)])
        self.assertEqual(len(rawio._sigs_memmaps), 3)  # check that there are only 3 memmaps

        # Test Cheetah 6.4.1, with different sampling rates across ncs files.
        rawio = NeuralynxRawIO(self.get_local_path('neuralynx/Cheetah_v6.4.1dev/original_data'))
        rawio.parse_header()

        self.assertEqual(rawio._nb_segment, 1)
        seg_idx = 0

        self.assertEqual(rawio.signal_streams_count(), 3)
        self.assertListEqual(rawio._timestamp_limits, [(1614363777825263, 1614363778481169)])
        self.assertDictEqual(rawio._sigs_length[seg_idx], {('CSC1', '26'): 15872,
                                                           ('LFP4', '41'): 1024,
                                                           ('WE1', '33'): 512})
        self.assertListEqual(rawio._signal_limits, [(1614363777.825263, 1614363778.481169)])
        # check that there are only 3 memmaps
        self.assertEqual(len(rawio._sigs_memmaps[seg_idx]), 3)

    def test_single_file_mode(self):
        """
        Tests reading of single files.
        """

        # test single analog signal channel
        fname = self.get_local_path('neuralynx/Cheetah_v5.6.3/original_data/CSC1.ncs')
        rawio = NeuralynxRawIO(filename=fname)
        rawio.parse_header()

        self.assertEqual(rawio._nb_segment, 2)
        self.assertEqual(len(rawio.ncs_filenames), 1)
        self.assertEqual(len(rawio.nev_filenames), 0)
        sigHdrs = rawio.header['signal_channels']
        self.assertEqual(sigHdrs.size, 1)
        self.assertEqual(sigHdrs[0][0], 'CSC1')
        self.assertEqual(sigHdrs[0][1], '58')
        self.assertEqual(len(rawio.header['spike_channels']), 0)
        self.assertEqual(len(rawio.header['event_channels']), 0)

        # test one single electrode channel
        fname = self.get_local_path('neuralynx/Cheetah_v5.5.1/original_data/STet3a.nse')
        rawio = NeuralynxRawIO(filename=fname)
        rawio.parse_header()

        self.assertEqual(rawio._nb_segment, 1)
        self.assertEqual(len(rawio.ncs_filenames), 0)
        self.assertEqual(len(rawio.nev_filenames), 0)
        seHdrs = rawio.header['spike_channels']
        self.assertEqual(len(seHdrs), 1)
        self.assertEqual(seHdrs[0][0], 'chSTet3a#8#0')
        self.assertEqual(seHdrs[0][1], '0')
        self.assertEqual(len(rawio.header['signal_channels']), 0)
        self.assertEqual(len(rawio.header['event_channels']), 0)

    def test_exclude_filenames(self):
        # exclude single ncs file from session
        dname = self.get_local_path('neuralynx/Cheetah_v5.6.3/original_data/')
        rawio = NeuralynxRawIO(dirname=dname, exclude_filename='CSC2.ncs')
        rawio.parse_header()

        self.assertEqual(rawio._nb_segment, 2)
        self.assertEqual(len(rawio.ncs_filenames), 1)
        self.assertEqual(len(rawio.nev_filenames), 1)
        sigHdrs = rawio.header['signal_channels']
        self.assertEqual(sigHdrs.size, 1)
        self.assertEqual(sigHdrs[0][0], 'CSC1')
        self.assertEqual(sigHdrs[0][1], '58')
        self.assertEqual(len(rawio.header['spike_channels']), 8)
        self.assertEqual(len(rawio.header['event_channels']), 2)

        # exclude multiple files from session
        rawio = NeuralynxRawIO(dirname=dname, exclude_filename=['Events.nev', 'CSC2.ncs'])
        rawio.parse_header()

        self.assertEqual(rawio._nb_segment, 2)
        self.assertEqual(len(rawio.ncs_filenames), 1)
        self.assertEqual(len(rawio.nev_filenames), 0)
        sigHdrs = rawio.header['signal_channels']
        self.assertEqual(sigHdrs.size, 1)
        self.assertEqual(sigHdrs[0][0], 'CSC1')
        self.assertEqual(sigHdrs[0][1], '58')
        self.assertEqual(len(rawio.header['spike_channels']), 8)
        self.assertEqual(len(rawio.header['event_channels']), 0)

class TestNcsRecordingType(TestNeuralynxRawIO, unittest.TestCase):
    """
    Test of decoding of NlxHeader for type of recording.
    """
    entities_to_test = []

    ncsTypeTestFiles = [
        ('neuralynx/Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs', 'PRE4'),
        ('neuralynx/Cheetah_v5.4.0/original_data/CSC5_trunc.Ncs', 'DIGITALLYNX'),
        ('neuralynx/Cheetah_v5.5.1/original_data/STet3a.nse', 'DIGITALLYNXSX'),
        ('neuralynx/Cheetah_v5.5.1/original_data/Tet3a.ncs', 'DIGITALLYNXSX'),
        ('neuralynx/Cheetah_v5.6.3/original_data/CSC1.ncs', 'DIGITALLYNXSX'),
        ('neuralynx/Cheetah_v5.6.3/original_data/TT1.ntt', 'DIGITALLYNXSX'),
        ('neuralynx/Cheetah_v5.7.4/original_data/CSC1.ncs', 'DIGITALLYNXSX'),
        ('neuralynx/Cheetah_v6.3.2/incomplete_blocks/CSC1_reduced.ncs', 'DIGITALLYNXSX')]

    def test_recording_types(self):

        for typeTest in self.ncsTypeTestFiles:

            filename = self.get_local_path(typeTest[0])
            hdr = NlxHeader(filename)
            self.assertEqual(hdr.type_of_recording(), typeTest[1])


class TestNcsSectionsFactory(TestNeuralynxRawIO, unittest.TestCase):
    """
    Test building NcsBlocks for files of different revisions.
    """
    entities_to_test = []

    def test_ncsblocks_partial(self):
        filename = self.get_local_path('neuralynx/Cheetah_v6.3.2/incomplete_blocks/CSC1_reduced.ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        self.assertEqual(data0.shape[0], 6690)
        self.assertEqual(data0['timestamp'][6689], 8515800549)  # timestamp of last record

        hdr = NlxHeader(filename)
        nb = NcsSectionsFactory.build_for_ncs_file(data0, hdr)
        self.assertEqual(nb.sampFreqUsed, 32000.012813673042)
        self.assertEqual(nb.microsPerSampUsed, 31.249987486652431)
        self.assertListEqual([blk.startRec for blk in nb.sects], [0, 1190, 4937])
        self.assertListEqual([blk.endRec for blk in nb.sects], [1189, 4936, 6689])

    def test_build_given_actual_frequency(self):

        # Test early files where the frequency listed in the header is
        # floor(1e6/(actual number of microseconds between samples)
        filename = self.get_local_path('neuralynx/Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        ncsBlocks = NcsSections()
        ncsBlocks.sampFreqUsed = 1 / (35e-6)
        ncsBlocks.microsPerSampUsed = 35
        ncsBlocks = NcsSectionsFactory._buildGivenActualFrequency(data0, ncsBlocks.sampFreqUsed,
                                                                  27789)
        self.assertEqual(len(ncsBlocks.sects), 1)
        self.assertEqual(ncsBlocks.sects[0].startRec, 0)
        self.assertEqual(ncsBlocks.sects[0].endRec, 9)

    def test_build_using_header_and_scanning(self):

        # Test early files where the frequency listed in the header is
        # floor(1e6/(actual number of microseconds between samples)
        filename = self.get_local_path('neuralynx/Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs')
        hdr = NlxHeader(filename)
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        nb = NcsSectionsFactory.build_for_ncs_file(data0, hdr)

        self.assertEqual(nb.sampFreqUsed, 1 / 35e-6)
        self.assertEqual(nb.microsPerSampUsed, 35)
        self.assertEqual(len(nb.sects), 1)
        self.assertEqual(nb.sects[0].startRec, 0)
        self.assertEqual(nb.sects[0].endRec, 9)

        # test Cheetah 5.5.1, which is DigitalLynxSX and has two blocks of records
        # with a fairly large gap
        filename = self.get_local_path('neuralynx/Cheetah_v5.5.1/original_data/Tet3a.ncs')
        hdr = NlxHeader(filename)
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        nb = NcsSectionsFactory.build_for_ncs_file(data0, hdr)
        self.assertEqual(nb.sampFreqUsed, 32000)
        self.assertEqual(nb.microsPerSampUsed, 31.25)
        self.assertEqual(len(nb.sects), 2)
        self.assertListEqual([blk.startRec for blk in nb.sects], [0, 2498])
        self.assertListEqual([blk.endRec for blk in nb.sects], [2497, 3331])

    def test_block_start_and_end_times(self):
        # digitallynxsx version to exercise the _parseForMaxGap function with multiple blocks
        filename = self.get_local_path('neuralynx/Cheetah_v6.3.2/incomplete_blocks/CSC1_reduced.ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr = NlxHeader(filename)
        nb = NcsSectionsFactory.build_for_ncs_file(data0, hdr)
        self.assertListEqual([blk.startTime for blk in nb.sects], [8408806811, 8427832053,
                                                                   8487768561])
        self.assertListEqual([blk.endTime for blk in nb.sects], [8427831990, 8487768498,
                                                                 8515816549])

        # digitallynxsx with single block of records to exercise path in _buildForMaxGap
        filename = self.get_local_path('neuralynx/Cheetah_v1.1.0/original_data/CSC67_trunc.Ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr = NlxHeader(filename)
        nb = NcsSectionsFactory.build_for_ncs_file(data0, hdr)
        self.assertEqual(len(nb.sects), 1)
        self.assertEqual(nb.sects[0].startTime, 253293161778)
        self.assertEqual(nb.sects[0].endTime, 253293349278)

        # PRE4 version with single block of records to exercise path in _buildGivenActualFrequency
        filename = self.get_local_path('neuralynx/Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr = NlxHeader(filename)
        nb = NcsSectionsFactory.build_for_ncs_file(data0, hdr)
        self.assertEqual(len(nb.sects), 1)
        self.assertEqual(nb.sects[0].startTime, 266982936)
        self.assertEqual(nb.sects[0].endTime, 267162136)

        # BML style with two blocks of records and one partially filled record to exercise
        # _parseGivenActualFrequency
        filename = self.get_local_path(
            'neuralynx/BML_unfilledsplit/original_data/unfilledSplitRecords.Ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr = NlxHeader(filename)
        nb = NcsSectionsFactory.build_for_ncs_file(data0, hdr)
        self.assertEqual(len(nb.sects), 2)
        self.assertListEqual([blk.startTime for blk in nb.sects], [1837623129, 6132625241])
        self.assertListEqual([blk.endTime for blk in nb.sects], [1837651009, 6132642649])

    def test_block_verify(self):
        # check that file verifies against itself for single block
        filename = self.get_local_path('neuralynx/Cheetah_v4.0.2/original_data/CSC14_trunc.Ncs')
        data0 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr0 = NlxHeader(filename)
        nb0 = NcsSectionsFactory.build_for_ncs_file(data0, hdr0)

        self.assertTrue(NcsSectionsFactory._verifySectionsStructure(data0, nb0))

        # check that fails against file with two blocks
        filename = self.get_local_path(
            'neuralynx/BML_unfilledsplit/original_data/unfilledSplitRecords.Ncs')
        data1 = np.memmap(filename, dtype=NeuralynxRawIO._ncs_dtype, mode='r',
                          offset=NlxHeader.HEADER_SIZE)
        hdr1 = NlxHeader(filename)
        nb1 = NcsSectionsFactory.build_for_ncs_file(data1, hdr1)

        self.assertFalse(NcsSectionsFactory._verifySectionsStructure(data1, nb0))

        # check that two blocks verify against self
        self.assertTrue(NcsSectionsFactory._verifySectionsStructure(data1, nb1))


class TestNcsSections(TestNeuralynxRawIO, unittest.TestCase):
    """
    Test building NcsBlocks for files of different revisions.
    """
    entities_to_test = []

    def test_equality(self):
        ns0 = NcsSections()
        ns1 = NcsSections()

        ns0.microsPerSampUsed = 1
        ns1.microsPerSampUsed = 1
        ns0.sampFreqUsed = 300
        ns1.sampFreqUsed = 300

        self.assertEqual(ns0, ns1)

        # add sections
        ns0.sects = [NcsSection(0, 0, 100, 100, 10)]
        ns1.sects = [NcsSection(0, 0, 100, 100, 10)]

        self.assertEqual(ns0, ns1)

        # check inequality for different attributes
        # different number of sections
        ns0.sects.append(NcsSection(0, 0, 100, 100, 10))
        self.assertNotEqual(ns0, ns1)

        # different section attributes
        ns0.sects = [NcsSection(0, 0, 200, 200, 10)]
        self.assertNotEqual(ns0, ns1)

        # different attributes
        ns0.sampFreqUsed = 400
        self.assertNotEqual(ns0, ns1)


if __name__ == "__main__":
    unittest.main()
