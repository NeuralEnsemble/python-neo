# encoding: utf-8
"""
Tests of the PyNNBinaryIO class


"""

from __future__ import with_statement, division
import numpy
import quantities as pq
import os
try:
    import unittest2 as unittest
except ImportError:
    import unittest
from neo.core import Segment, AnalogSignal, SpikeTrain
from neo.io import PyNNBinaryIO
from neo.test.tools import assert_arrays_equal

NCELLS = 5


def write_test_file(filename, variable='v', check=False):
    metadata = {
        'size': NCELLS,
        'first_index': 0,
        'first_id': 0,
        'n': 505,
        'variable': variable,
        'last_id': 4,
        'last_index': 5,
        'dt': 0.1,
        'label': "population0",
    }
    data = numpy.empty((505, 2))
    for id in range(NCELLS):
        data[id*101:(id+1)*101, 0] = numpy.arange(id, id+101, dtype=float) # signal
        data[id*101:(id+1)*101, 1] = id*numpy.ones((101,), dtype=float) # id
    metadata_array = numpy.array(list(metadata.items()))
    numpy.savez(filename, data=data, metadata=metadata_array)
    if check:
        data1, metadata1 = read_test_file(filename)
        assert metadata == metadata1, "%s != %s" % (metadata, metadata1)
        assert data.shape == data1.shape == (505, 2), "%s, %s, (505, 2)" % (data.shape, data1.shape)
        assert (data == data1).all()
        assert metadata["n"] == 505

def read_test_file(filename):
    contents = numpy.load(filename)
    data = contents["data"]
    metadata = {}
    for name,value in contents['metadata']:
        try:
            metadata[name] = eval(value)
        except Exception:
            metadata[name] = value
    return data, metadata


class TestPyNNBinaryIO_Signals(unittest.TestCase):
    
    def setUp(self):
        self.test_file = "test_file.npz" 
        write_test_file(self.test_file, "v")
    
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_read_segment_containing_analogsignals_using_eager_cascade(self):
        # eager == not lazy
        io = PyNNBinaryIO(self.test_file)
        segment = io.read_segment(lazy=False, cascade=True)
        self.assertIsInstance(segment, Segment)
        self.assertEqual(len(segment._analogsignals), NCELLS)
        as0 = segment._analogsignals[0]
        self.assertIsInstance(as0, AnalogSignal)
        assert_arrays_equal(as0,
                            AnalogSignal(numpy.arange(0, 101, dtype=float),
                                         sampling_period=0.1*pq.ms,
                                         t_start=0*pq.s,
                                         units=pq.mV))
        as4 = segment._analogsignals[4]
        self.assertIsInstance(as4, AnalogSignal)
        assert_arrays_equal(as4,
                            AnalogSignal(numpy.arange(4, 105, dtype=float),
                                         sampling_period=0.1*pq.ms,
                                         t_start=0*pq.s,
                                         units=pq.mV))
        # test annotations (stuff from file metadata)

    def test_read_analogsignal_using_eager(self):
        io = PyNNBinaryIO(self.test_file)
        as3 = io.read_analogsignal(lazy=False, channel_index=3)
        self.assertIsInstance(as3, AnalogSignal)
        assert_arrays_equal(as3,
                            AnalogSignal(numpy.arange(3, 104, dtype=float),
                                         sampling_period=0.1*pq.ms,
                                         t_start=0*pq.s,
                                         units=pq.mV))
        # should test annotations: 'channel_index', etc.


class TestPyNNBinaryIO_Spikes(unittest.TestCase):
    
    def setUp(self):
        self.test_file = "test_file_spikes.npz" 
        write_test_file(self.test_file, "spikes")
    
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_read_spiketrain_using_eager(self):
        io = PyNNBinaryIO(self.test_file)
        st3 = io.read_spiketrain(lazy=False, channel_index=3)
        self.assertIsInstance(st3, SpikeTrain)
        assert_arrays_equal(st3,
                            SpikeTrain(numpy.arange(3, 104, dtype=float),
                                       t_start=0*pq.s,
                                       units=pq.ms))
        # should test annotations: 'channel_index', etc.


if __name__ == '__main__':
    write_test_file("test_file.npz", check=True)
    os.remove("test_file.npz")
    unittest.main()
