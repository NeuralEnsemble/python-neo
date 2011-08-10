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
from neo.core import Segment, AnalogSignal
from neo.io import PyNNBinaryIO
from neo.test.tools import assert_arrays_equal

NCELLS = 5


def write_test_file(filename, check=False):
    metadata = {
        'size': NCELLS,
        'first_index': 0,
        'first_id': 0,
        'n': 505,
        'variable': "v",
        'last_id': 4,
        'last_index': 5,
        'dt': 0.1,
        'label': "population0",
    }
    data = numpy.empty((505, 2))
    for id in range(NCELLS):
        data[id*101:(id+1)*101, 0] = numpy.arange(id, id+101, dtype=float)
        data[id*101:(id+1)*101, 1] = id*numpy.ones((101,), dtype=float)
    metadata_array = numpy.array(metadata.items())
    with open(filename, 'w') as f:
        numpy.savez(f, data=data, metadata=metadata_array)
    if check:
        data1, metadata1 = read_test_file(filename)
        assert metadata == metadata1, "%s != %s" % (metadata, metadata1)
        assert data.shape == data1.shape == (505, 2), "%s, %s, (505, 2)" % (data.shape, data1.shape)
        assert (data == data1).all()
        assert metadata["n"] == 505

def read_test_file(filename):
    with open(filename, 'r') as f:
        contents = numpy.load(f)
        data = contents["data"]
        metadata = {}
        for name,value in contents['metadata']:
            try:
                metadata[name] = eval(value)
            except Exception:
                metadata[name] = value
    return data, metadata


class TestPyNNBinaryIO(unittest.TestCase):
    
    def setUp(self):
        self.test_file = "test_file.npz" 
        write_test_file(self.test_file)
    
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_read_segment_containing_analog_signals_using_eager_cascade(self):
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





if __name__ == '__main__':
    write_test_file("test_file.npz", check=True)
    os.remove("test_file.npz")
    unittest.main()
