"""
Tests of neo.io.klustakwikio
"""

from pathlib import Path
import shutil

import unittest

import numpy as np
import quantities as pq

import neo
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.tools import assert_arrays_almost_equal
from neo.io.klustakwikio import KlustaKwikIO


class KlustaKwikTests(BaseTestIO, unittest.TestCase):
    ioclass = KlustaKwikIO
    entities_to_download = [
        'klustakwik'
    ]
    entities_to_test = [
        'klustakwik/test2/base',
        'klustakwik/test2/base2',
    ]

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(KlustaKwikTests, cls).setUpClass(*args, **kwargs)
        dirname = Path(cls.get_local_path('klustakwik'))

        cls.session1 = dirname / 'test1'
        cls.session2 = dirname / 'test2'
        cls.session3 = dirname / 'test3'
        cls.tmp_session = dirname / 'tmp_session'
        cls.tmp_session.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_session, ignore_errors=True)
        self.tmp_session.mkdir(exist_ok=True)

    def test_load_by_basename(self):
        """Tests that files can be loaded by basename"""
        kio = KlustaKwikIO(dirname=self.session1 / 'basename')
        fetfiles = kio._fp.read_filenames('fet')

        self.assertEqual(len(fetfiles), 2)
        self.assertEqual(Path(fetfiles[0]).absolute(), self.session1 / 'basename.fet.0')
        self.assertEqual(Path(fetfiles[1]).absolute(), self.session1 / 'basename.fet.1')

    # def test_load_without_basename(self):
    #     """Tests that files are loaded even without basename"""
    #     pass
    #
    #     # this test is in flux, should probably have it default to
    #     # basename = os.path.split(dirname)[1] when dirname is a directory)
    #     kio = KlustaKwikIO(dirname=self.session1)
    #     fetfiles = kio._fp.read_filenames('fet')
    #
    #     # It will just choose one of the two basenames, depending on which
    #     # is first, so just assert that it did something without error.
    #     self.assertNotEqual(len(fetfiles), 0)

    def test_load_by_basename_2(self):
        """Tests that files can be loaded by basename2"""
        kio = KlustaKwikIO(dirname=self.session1 / 'basename2')
        clufiles = kio._fp.read_filenames('clu')

        self.assertEqual(len(clufiles), 1)
        self.assertEqual(Path(clufiles[1]).absolute(), self.session1 /'basename2.clu.1')

    def test_read_data(self):
        """Tests that data and metadata are read correctly"""
        kio = KlustaKwikIO(dirname=self.session2 / 'base',
                           sampling_rate=1000.)
        block = kio.read()[0]
        seg = block.segments[0]
        self.assertEqual(len(seg.spiketrains), 4)

        for st in seg.spiketrains:
            self.assertEqual(st.units, np.array(1.0) * pq.s)
            self.assertEqual(st.t_start, 0.0)

        self.assertEqual(seg.spiketrains[0].name, 'unit 1 from group 0')
        self.assertEqual(seg.spiketrains[0].annotations['cluster'], 1)
        self.assertEqual(seg.spiketrains[0].annotations['group'], 0)
        self.assertTrue(np.all(seg.spiketrains[0].times == np.array([.100,
                                                                     .200])))

        self.assertEqual(seg.spiketrains[1].name, 'unit 2 from group 0')
        self.assertEqual(seg.spiketrains[1].annotations['cluster'], 2)
        self.assertEqual(seg.spiketrains[1].annotations['group'], 0)
        self.assertEqual(seg.spiketrains[1].t_start, 0.0)
        self.assertTrue(np.all(seg.spiketrains[1].times == np.array([.305])))

        self.assertEqual(seg.spiketrains[2].name, 'unit -1 from group 1')
        self.assertEqual(seg.spiketrains[2].annotations['cluster'], -1)
        self.assertEqual(seg.spiketrains[2].annotations['group'], 1)
        self.assertEqual(seg.spiketrains[2].t_start, 0.0)
        self.assertTrue(np.all(seg.spiketrains[2].times == np.array([.253])))

        self.assertEqual(seg.spiketrains[3].name, 'unit 2 from group 1')
        self.assertEqual(seg.spiketrains[3].annotations['cluster'], 2)
        self.assertEqual(seg.spiketrains[3].annotations['group'], 1)
        self.assertEqual(seg.spiketrains[3].t_start, 0.0)
        self.assertTrue(np.all(seg.spiketrains[3].times == np.array([.050,
                                                                     .152])))

    def test_default_cluster_id_0(self):
        """Checks that cluster id autosets to 0 without clu file"""
        kio = KlustaKwikIO(dirname=self.session2 / 'base2',
                           sampling_rate=1000.)
        block = kio.read()[0]
        seg = block.segments[0]
        self.assertEqual(len(seg.spiketrains), 1)
        self.assertEqual(seg.spiketrains[0].name, 'unit 0 from group 5')
        self.assertEqual(seg.spiketrains[0].annotations['cluster'], 0)
        self.assertEqual(seg.spiketrains[0].annotations['group'], 5)
        self.assertEqual(seg.spiketrains[0].t_start, 0.0)
        self.assertTrue(np.all(seg.spiketrains[0].times == np.array([0.026,
                                                                     0.122,
                                                                     0.228])))

    def test_write_clu_and_fet(self):
        """Create clu and fet files based on spiketrains in a block.

        Checks that
            Files are created
            Converted to samples correctly
            Missing sampling rate are taken from IO reader default
            Spiketrains without cluster info are assigned to cluster 0
            Spiketrains across segments are concatenated
        """
        block = neo.Block()
        segment = neo.Segment()
        segment2 = neo.Segment()
        block.segments.append(segment)
        block.segments.append(segment2)

        # Fake spiketrain 1, will be sorted
        st1 = neo.SpikeTrain(times=[.002, .004, .006], units='s', t_stop=1.)
        st1.annotations['cluster'] = 0
        st1.annotations['group'] = 0
        segment.spiketrains.append(st1)

        # Fake spiketrain 1B, on another segment. No group specified,
        # default is 0.
        st1B = neo.SpikeTrain(times=[.106], units='s', t_stop=1.)
        st1B.annotations['cluster'] = 0
        segment2.spiketrains.append(st1B)

        # Fake spiketrain 2 on same group, no sampling rate specified
        st2 = neo.SpikeTrain(times=[.001, .003, .011], units='s', t_stop=1.)
        st2.annotations['cluster'] = 1
        st2.annotations['group'] = 0
        segment.spiketrains.append(st2)

        # Fake spiketrain 3 on new group, with different sampling rate
        st3 = neo.SpikeTrain(times=[.05, .09, .10], units='s', t_stop=1.)
        st3.annotations['cluster'] = -1
        st3.annotations['group'] = 1
        segment.spiketrains.append(st3)

        # Fake spiketrain 4 on new group, without cluster info
        st4 = neo.SpikeTrain(times=[.005, .009], units='s', t_stop=1.)
        st4.annotations['group'] = 2
        segment.spiketrains.append(st4)

        # Create writer with default sampling rate
        kio = KlustaKwikIO(dirname=self.tmp_session / 'base1',
                           sampling_rate=1000.)
        kio.write_block(block)

        # Check files were created
        for fn in ['.fet.0', '.fet.1', '.clu.0', '.clu.1']:
            self.assertTrue((self.tmp_session / ('base1' + fn)).exists())

        # Check files contain correct content
        # Spike times on group 0
        with open(self.tmp_session / 'base1.fet.0', mode='r') as f:
            data = f.readlines()
        data = [int(d) for d in data]
        self.assertEqual(data, [0, 2, 4, 6, 1, 3, 11, 106])

        # Clusters on group 0
        with open(self.tmp_session / 'base1.clu.0', mode='r') as f:
            data = f.readlines()
        data = [int(d) for d in data]
        self.assertEqual(data, [2, 0, 0, 0, 1, 1, 1, 0])

        # Spike times on group 1
        with open(self.tmp_session / 'base1.fet.1', mode='r') as f:
            data = f.readlines()
        data = [int(d) for d in data]
        self.assertEqual(data, [0, 50, 90, 100])

        # Clusters on group 1
        with open(self.tmp_session / 'base1.clu.1') as f:
            data = f.readlines()
        data = [int(d) for d in data]
        self.assertEqual(data, [1, -1, -1, -1])

        # Spike times on group 2
        with open(self.tmp_session / 'base1.fet.2') as f:
            data = f.readlines()
            data = [int(d) for d in data]
        self.assertEqual(data, [0, 5, 9])

        # Clusters on group 2
        with open(self.tmp_session / 'base1.clu.2') as f:
            data = f.readlines()
        data = [int(d) for d in data]
        self.assertEqual(data, [1, 0, 0])

    def test_write_clu_and_fet_1(self):
        """Create clu and fet files based on spiketrains in a block.

        Checks that
            Files are created
            Converted to samples correctly
            Missing sampling rate are taken from IO reader default
            Spiketrains without cluster info are assigned to cluster 0
            Spiketrains across segments are concatenated
        """
        block = neo.Block()
        segment = neo.Segment()
        segment2 = neo.Segment()
        block.segments.append(segment)
        block.segments.append(segment2)

        # Fake spiketrain 1
        st1 = neo.SpikeTrain(times=[.002, .004, .006], units='s', t_stop=1.)
        st1.annotations['cluster'] = 0
        st1.annotations['group'] = 0
        wff = np.array([
            [11.3, 0.2],
            [-0.3, 12.3],
            [3.0, -2.5]])
        st1.annotations['waveform_features'] = wff
        segment.spiketrains.append(st1)

        # Create writer
        kio = KlustaKwikIO(dirname=self.tmp_session / 'base2',
                           sampling_rate=1000.)
        kio.write_block(block)

        # Check files were created
        for fn in ['.fet.0', '.clu.0']:
            self.assertTrue(self.tmp_session / ('base2' + fn))

        # Check files contain correct content
        with open(self.tmp_session / 'base2.fet.0') as fi:
            # first line is nbFeatures
            self.assertEqual(fi.readline(), '2\n')

            # Now check waveforms and times are same
            data = fi.readlines()
            new_wff = []
            new_times = []
            for line in data:
                line_split = line.split()
                new_wff.append([float(val) for val in line_split[:-1]])
                new_times.append(int(line_split[-1]))
            self.assertEqual(new_times, [2, 4, 6])
            assert_arrays_almost_equal(wff, np.array(new_wff), .00001)

        # Clusters on group 0
        with open(self.tmp_session / 'base2.clu.0') as fi:
            data = fi.readlines()
            data = [int(d) for d in data]
            self.assertEqual(data, [1, 0, 0, 0])

            # Now read the features and test same
            block = kio.read_block()
            train = block.segments[0].spiketrains[0]
            assert_arrays_almost_equal(wff, train.annotations['waveform_features'],
                                       .00001)


if __name__ == '__main__':
    unittest.main()
