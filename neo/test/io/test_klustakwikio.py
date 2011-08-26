import unittest
import neo.io.klustakwikio
import os.path
import numpy as np
import quantities as pq
import glob

class testFilenameParser(unittest.TestCase):
    """Tests that filenames can be loaded with or without basename.
    
    The test directory contains two basenames and some decoy files with
    malformed group numbers."""
    def test1(self):
        """Tests that files can be loaded by basename"""
        dirname = os.path.normpath('./test/test1')
        kio = neo.io.KlustaKwikIO(filename=dirname, basename='basename')
        fetfiles = kio._fp.read_filenames('fet')
        
        self.assertEqual(len(fetfiles), 2)
        self.assertEqual(fetfiles[0], os.path.join(dirname, 'basename.fet.0'))
        self.assertEqual(fetfiles[1], os.path.join(dirname, 'basename.fet.1'))

    def test2(self):
        """Tests that files are loaded even without basename"""
        dirname = os.path.normpath('./test/test1')
        kio = neo.io.KlustaKwikIO(filename=dirname)
        fetfiles = kio._fp.read_filenames('fet')
        
        # It will just choose one of the two basenames, depending on which
        # is first, so just assert that it did something without error.
        self.assertNotEqual(len(fetfiles), 0)

    def test3(self):
        """Tests that files can be loaded by basename2"""
        dirname = os.path.normpath('./test/test1')
        kio = neo.io.KlustaKwikIO(filename=dirname, basename='basename2')
        clufiles = kio._fp.read_filenames('clu')
        
        self.assertEqual(len(clufiles), 1)
        self.assertEqual(clufiles[1], os.path.join(dirname, 'basename2.clu.1'))



class testRead(unittest.TestCase):
    """Tests that data can be read from KlustaKwik files"""
    def test1(self):
        """Tests that data and metadata are read correctly"""
        dirname = os.path.normpath('./test/test2')
        kio = neo.io.KlustaKwikIO(filename=dirname, sampling_rate=1000., 
            basename='base')
        block = kio.read()
        seg = block._segments[0]
        self.assertEqual(len(seg._spiketrains), 4)
        
        for st in seg._spiketrains:
            self.assertEqual(st.units, np.array(1.0) * pq.s)
            self.assertEqual(st.t_start, 0.0)
        
        
        self.assertEqual(seg._spiketrains[0].name, 'unit 1 from group 0')
        self.assertEqual(seg._spiketrains[0]._annotations['cluster'], 1)
        self.assertEqual(seg._spiketrains[0]._annotations['group'], 0)
        self.assertTrue(np.all(seg._spiketrains[0].times == np.array(
            [.100, .200])))

        self.assertEqual(seg._spiketrains[1].name, 'unit 2 from group 0')
        self.assertEqual(seg._spiketrains[1]._annotations['cluster'], 2)
        self.assertEqual(seg._spiketrains[1]._annotations['group'], 0)
        self.assertEqual(seg._spiketrains[1].t_start, 0.0)
        self.assertTrue(np.all(seg._spiketrains[1].times == np.array([.305])))
        
        self.assertEqual(seg._spiketrains[2].name, 'unit -1 from group 1')
        self.assertEqual(seg._spiketrains[2]._annotations['cluster'], -1)
        self.assertEqual(seg._spiketrains[2]._annotations['group'], 1)
        self.assertEqual(seg._spiketrains[2].t_start, 0.0)
        self.assertTrue(np.all(seg._spiketrains[2].times == np.array([.253])))
        
        self.assertEqual(seg._spiketrains[3].name, 'unit 2 from group 1')
        self.assertEqual(seg._spiketrains[3]._annotations['cluster'], 2)
        self.assertEqual(seg._spiketrains[3]._annotations['group'], 1)
        self.assertEqual(seg._spiketrains[3].t_start, 0.0)
        self.assertTrue(np.all(seg._spiketrains[3].times == np.array(
            [.050, .152])))        
    
    def test2(self):
        """Checks that cluster id autosets to 0 without clu file"""
        dirname = os.path.normpath('./test/test2')
        kio = neo.io.KlustaKwikIO(filename=dirname, sampling_rate=1000., 
            basename='base2')
        block = kio.read()
        seg = block._segments[0]
        self.assertEqual(len(seg._spiketrains), 1)
        self.assertEqual(seg._spiketrains[0].name, 'unit 0 from group 5')
        self.assertEqual(seg._spiketrains[0]._annotations['cluster'], 0)
        self.assertEqual(seg._spiketrains[0]._annotations['group'], 5)        
        self.assertEqual(seg._spiketrains[0].t_start, 0.0)
        self.assertTrue(np.all(seg._spiketrains[0].times == np.array(
            [0.026, 0.122, 0.228])))

class testWrite(unittest.TestCase):
    def test1(self):
        """Create clu and fet files based on spiketrains in a block.
        
        Checks that
            Files are created
            Spike times are sorted
            Converted to samples correctly
            Missing sampling rate are taken from IO reader default        
            Spiketrains without cluster info are assigned to cluster 0
            Spiketrains across segments are concatenated
        """
        block = neo.Block()
        segment = neo.Segment()
        segment2 = neo.Segment()
        block._segments.append(segment)
        block._segments.append(segment2)
        
        # Fake spiketrain 1, will be sorted
        st1 = neo.SpikeTrain(times=[.006, .002, .004], units='s', 
            sampling_rate=1000.)
        st1._annotations['cluster'] = 0
        st1._annotations['group'] = 0
        segment._spiketrains.append(st1)
        
        # Fake spiketrain 1B, on another segment. No group specified,
        # default is 0.
        st1B = neo.SpikeTrain(times=[.106], units='s', sampling_rate=1000.)
        st1B._annotations['cluster'] = 0        
        segment2._spiketrains.append(st1B)
        
        # Fake spiketrain 2 on same group, no sampling rate specified
        st2 = neo.SpikeTrain(times=[.001, .003, .011], units='s')
        st2._annotations['cluster'] = 1
        st2._annotations['group'] = 0
        segment._spiketrains.append(st2)
        
        # Fake spiketrain 3 on new group, with different sampling rate
        st3 = neo.SpikeTrain(times=[.05, .09, .10], units='s', 
            sampling_rate=100.)
        st3._annotations['cluster'] = -1
        st3._annotations['group'] = 1
        segment._spiketrains.append(st3)
        
        # Fake spiketrain 4 on new group, without cluster info
        st4 = neo.SpikeTrain(times=[.005, .009], units='s')
        st4._annotations['group'] = 2
        segment._spiketrains.append(st4)
        
        # Create empty directory for writing
        dirname = os.path.normpath('./test/test3')
        delete_test_session()
        
        # Create writer with default sampling rate
        kio = neo.io.KlustaKwikIO(filename=dirname, basename='base1',
            sampling_rate=1000.)
        kio.write_block(block)
        
        # Check files were created
        for fn in ['.fet.0', '.fet.1', '.clu.0', '.clu.1']:
            self.assertTrue(os.path.exists(os.path.join(dirname,
                'base1' + fn)))
        
        # Check files contain correct content
        # Spike times on group 0
        data = file(os.path.join(dirname, 'base1.fet.0')).readlines()
        data = [int(d) for d in data]
        self.assertTrue(data == [0, 2, 4, 6, 1, 3, 11, 106])
        
        # Clusters on group 0
        data = file(os.path.join(dirname, 'base1.clu.0')).readlines()
        data = [int(d) for d in data]
        self.assertTrue(data == [2, 0, 0, 0, 1, 1, 1, 0])
        
        # Spike times on group 1
        data = file(os.path.join(dirname, 'base1.fet.1')).readlines()
        data = [int(d) for d in data]
        self.assertTrue(data == [0, 5, 9, 10])
        
        # Clusters on group 1
        data = file(os.path.join(dirname, 'base1.clu.1')).readlines()
        data = [int(d) for d in data]
        self.assertTrue(data == [1, -1, -1, -1])
        
        # Spike times on group 2
        data = file(os.path.join(dirname, 'base1.fet.2')).readlines()
        data = [int(d) for d in data]
        self.assertTrue(data == [0, 5, 9])
        
        # Clusters on group 2
        data = file(os.path.join(dirname, 'base1.clu.2')).readlines()
        data = [int(d) for d in data]
        self.assertTrue(data == [1, 0, 0])


def delete_test_session():
    for fi in glob.glob(os.path.join('./test/test3', '*')): os.remove(fi)
    

if __name__ == '__main__':
    unittest.main()
