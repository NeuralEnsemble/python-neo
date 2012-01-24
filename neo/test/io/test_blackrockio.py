
import unittest
import neo.io.blackrockio
import os
import numpy as np
import quantities as pq
import glob

from neo.test.io.common_io_test import BaseTestIO
from neo.io import tools
from neo.test.tools import assert_arrays_almost_equal
import struct



#~ class testRead(unittest.TestCase):
    #~ """Tests that data can be read from KlustaKwik files"""
    #~ def test1(self):
        #~ """Tests that data and metadata are read correctly"""
        #~ pass
    #~ def test2(self):
        #~ """Checks that cluster id autosets to 0 without clu file"""
        #~ pass
        #~ dirname = os.path.normpath('./files_for_tests/klustakwik/test2')
        #~ kio = neo.io.KlustaKwikIO(filename=os.path.join(dirname, 'base2'),
            #~ sampling_rate=1000.)
        #~ block = kio.read()
        #~ seg = block.segments[0]
        #~ self.assertEqual(len(seg.spiketrains), 1)
        #~ self.assertEqual(seg.spiketrains[0].name, 'unit 0 from group 5')
        #~ self.assertEqual(seg.spiketrains[0].annotations['cluster'], 0)
        #~ self.assertEqual(seg.spiketrains[0].annotations['group'], 5)        
        #~ self.assertEqual(seg.spiketrains[0].t_start, 0.0)
        #~ self.assertTrue(np.all(seg.spiketrains[0].times == np.array(
            #~ [0.026, 0.122, 0.228])))

class testWrite(unittest.TestCase):
    fn = os.path.join(os.path.dirname(__file__), 
        './files_for_tests/blackrock/test2/test.write.ns5')
    def test1(self):
        """Write data to binary file, then read it back in and verify"""
        # delete temporary file before trying to write to it
        if os.path.exists(self.fn): os.remove(self.fn)

        block = neo.Block()
        full_range = 234 * pq.mV
        
        # Create segment1 with analogsignals
        segment1 = neo.Segment()
        sig1 = neo.AnalogSignal([3,4,5], units='mV', channel_index=3,
            sampling_rate=30000.*pq.Hz)
        sig2 = neo.AnalogSignal([6,-4,-5], units='mV', channel_index=4,
            sampling_rate=30000.*pq.Hz)
        segment1.analogsignals.append(sig1)
        segment1.analogsignals.append(sig2)
        
        # Create segment2 with analogsignals
        segment2 = neo.Segment()
        sig3 = neo.AnalogSignal([-3,-4,-5], units='mV', channel_index=3,
            sampling_rate=30000.*pq.Hz)
        sig4 = neo.AnalogSignal([-6,4,5], units='mV', channel_index=4,
            sampling_rate=30000.*pq.Hz)
        segment2.analogsignals.append(sig3)
        segment2.analogsignals.append(sig4)        
        
        
        # Link segments to block
        block.segments.append(segment1)
        block.segments.append(segment2)
        
        # Create hardware view, and bijectivity
        #tools.populate_RecordingChannel(block)
        #print "problem happening"
        #print block.recordingchannelgroups[0].recordingchannels
        #print block.recordingchannelgroups[0].recordingchannels[0].analogsignals
        #tools.create_many_to_one_relationship(block)
        #print "here: "
        #print block.segments[0].analogsignals[0].recordingchannel
        
        # Chris I prefer that:
        #tools.finalize_block(block)
        tools.populate_RecordingChannel(block)
        tools.create_many_to_one_relationship(block)
        
        
        # Check that blackrockio is correctly extracting channel indexes
        self.assertEqual(neo.io.blackrockio.channel_indexes_in_segment(
            segment1), [3,4])
        self.assertEqual(neo.io.blackrockio.channel_indexes_in_segment(
            segment2), [3,4])
        
        # Create writer. Write block, then read back in.
        bio = neo.io.BlackrockIO(filename=self.fn, full_range=full_range)
        bio.write_block(block)
        fi = file(self.fn)
        
        # Text header
        self.assertEqual(fi.read(16), 'NEURALSG30 kS/s\x00')
        self.assertEqual(fi.read(8), '\x00\x00\x00\x00\x00\x00\x00\x00')
        
        # Integers: period, channel count, channel index1, channel index2
        self.assertEqual(struct.unpack('<4I', fi.read(16)), (1,2,3,4))

        # What should the signals be after conversion?
        conv = float(full_range) / 2**16
        sigs = np.array(\
            [np.concatenate((sig1,sig3)), np.concatenate((sig2, sig4))])
        sigs_converted = np.rint(sigs / conv).astype(np.int)

        # Check that each time point is the same
        for time_slc in sigs_converted.transpose():
            written_data = struct.unpack('<2h', fi.read(4))
            self.assertEqual(list(time_slc), list(written_data))
        
        # Check that we read to the end
        currentpos = fi.tell()
        fi.seek(0, 2)
        truelen = fi.tell()
        self.assertEqual(currentpos, truelen)
        fi.close()
        
        # Empty out test session again
        #~ delete_test_session()

class testRead(unittest.TestCase):
    fn = os.path.join(os.path.dirname(__file__), 
        './files_for_tests/blackrock/test2/test.ns5')
    def test1(self):
        """Read data into one big segment (default)"""
        full_range = 8192 * pq.mV
        bio = neo.io.BlackrockIO(filename=self.fn, full_range=full_range)
        block = bio.read_block(n_starts=[0], n_stops=[6])
        self.assertEqual(bio.header.Channel_Count, 2)
        self.assertEqual(bio.header.n_samples, 6)
        
        # Everything put in one segment
        self.assertEqual(len(block.segments), 1)
        seg = block.segments[0]
        self.assertEqual(len(seg.analogsignals), 2)
        
        assert_arrays_almost_equal(seg.analogsignals[0], 
            [3., 4., 5., -3., -4., -5.] * pq.mV, .0001)
        assert_arrays_almost_equal(seg.analogsignals[1], 
            [6., -4., -5., -6., 4., 5.] * pq.mV, .0001)

    def test2(self):
        """Read data into two segments instead of just one"""
        full_range = 8192 * pq.mV
        bio = neo.io.BlackrockIO(filename=self.fn, full_range=full_range)
        block = bio.read_block(n_starts=[0, 3], n_stops=[2, 6])
        self.assertEqual(bio.header.Channel_Count, 2)
        self.assertEqual(bio.header.n_samples, 6)
        
        # Everything in two segments
        self.assertEqual(len(block.segments), 2)
        
        # Test first seg
        seg = block.segments[0]
        self.assertEqual(len(seg.analogsignals), 2)        
        assert_arrays_almost_equal(seg.analogsignals[0], 
            [3., 4.] * pq.mV, .0001)
        assert_arrays_almost_equal(seg.analogsignals[1], 
            [6., -4.] * pq.mV, .0001)

        # Test second seg
        seg = block.segments[1]
        self.assertEqual(len(seg.analogsignals), 2)        
        assert_arrays_almost_equal(seg.analogsignals[0], 
            [-3., -4., -5.] * pq.mV, .0001)
        assert_arrays_almost_equal(seg.analogsignals[1], 
            [-6., 4., 5.] * pq.mV, .0001)


class CommonTests(BaseTestIO, unittest.TestCase ):
    ioclass = neo.io.BlackrockIO
    read_and_write_is_bijective = False
    
    # These are the files it tries to read and test for compliance
    files_to_test = [ 
        'test2/test.ns5'
        ]
    
    # Will fetch from g-node if they don't already exist locally
    # How does it know to do this before any of the other tests?
    files_to_download = [ 
        'test2/test.ns5'
        ]


#~ def delete_test_session():
    #~ """Removes all file in directory so we can test writing to it"""
    #~ for fi in glob.glob(os.path.join(
        #~ './files_for_tests/klustakwik/test3', '*')): 
        #~ os.remove(fi)
    

if __name__ == '__main__':
    unittest.main()
