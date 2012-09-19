# encoding: utf-8
"""
Tests of the PyNNNumpyIO and PyNNTextIO classes


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
from neo.io import PyNNNumpyIO, PyNNTextIO
from neo.test.tools import assert_arrays_equal, assert_file_contents_equal


NCELLS = 5


#TODO: common test fails.
from neo.test.io.common_io_test import BaseTestIO
#class CommonTestPyNNNumpyIO(BaseTestIO, unittest.TestCase):
#    ioclass = PyNNNumpyIO

class CommonTestPyNNTextIO(BaseTestIO, unittest.TestCase):
    ioclass = PyNNTextIO
    read_and_write_is_bijective = False




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
read_test_file.__test__ = False
 

class BaseTestPyNNIO(object):
    __test__ = False
    
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_write_segment(self):
        in_ = self.io_cls(self.test_file)
        write_test_file = "write_test.%s" % self.file_extension
        out = self.io_cls(write_test_file)
        out.write_segment(in_.read_segment(lazy=False, cascade=True))
        assert_file_contents_equal(self.test_file, write_test_file)
        if os.path.exists(write_test_file):
            os.remove(write_test_file)

    def build_test_data(self, variable='v'):
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
        if variable == 'v':
            metadata['units'] = 'mV'
        elif variable == 'spikes':
            metadata['units'] = 'ms'
        data = numpy.empty((505, 2))
        for i in range(NCELLS):
            data[i*101:(i+1)*101, 0] = numpy.arange(i, i+101, dtype=float) # signal
            data[i*101:(i+1)*101, 1] = i*numpy.ones((101,), dtype=float) # index
        return data, metadata
    build_test_data.__test__ = False


class BaseTestPyNNIO_Signals(BaseTestPyNNIO):

    def setUp(self):
        self.test_file = "test_file_v.%s" % self.file_extension
        self.write_test_file("v")

    def test_read_segment_containing_analogsignals_using_eager_cascade(self):
        # eager == not lazy
        io = self.io_cls(self.test_file)
        segment = io.read_segment(lazy=False, cascade=True)
        self.assertIsInstance(segment, Segment)
        self.assertEqual(len(segment.analogsignals), NCELLS)
        as0 = segment.analogsignals[0]
        self.assertIsInstance(as0, AnalogSignal)
        assert_arrays_equal(as0,
                            AnalogSignal(numpy.arange(0, 101, dtype=float),
                                         sampling_period=0.1*pq.ms,
                                         t_start=0*pq.s,
                                         units=pq.mV))
        as4 = segment.analogsignals[4]
        self.assertIsInstance(as4, AnalogSignal)
        assert_arrays_equal(as4,
                            AnalogSignal(numpy.arange(4, 105, dtype=float),
                                         sampling_period=0.1*pq.ms,
                                         t_start=0*pq.s,
                                         units=pq.mV))
        # test annotations (stuff from file metadata)

    def test_read_analogsignal_using_eager(self):
        io = self.io_cls(self.test_file)
        as3 = io.read_analogsignal(lazy=False, channel_index=3)
        self.assertIsInstance(as3, AnalogSignal)
        assert_arrays_equal(as3,
                            AnalogSignal(numpy.arange(3, 104, dtype=float),
                                         sampling_period=0.1*pq.ms,
                                         t_start=0*pq.s,
                                         units=pq.mV))
        # should test annotations: 'channel_index', etc.

    def test_read_spiketrain_should_fail_with_analogsignal_file(self):
        io = self.io_cls(self.test_file)
        self.assertRaises(TypeError, io.read_spiketrain, channel_index=0)

    


class BaseTestPyNNIO_Spikes(BaseTestPyNNIO):
    
    def setUp(self):
        self.test_file = "test_file_spikes.%s" % self.file_extension
        self.write_test_file("spikes")

    def test_read_segment_containing_spiketrains_using_eager_cascade(self):
        io = self.io_cls(self.test_file)
        segment = io.read_segment(lazy=False, cascade=True)
        self.assertIsInstance(segment, Segment)
        self.assertEqual(len(segment.spiketrains), NCELLS)
        st0 = segment.spiketrains[0]
        self.assertIsInstance(st0, SpikeTrain)
        assert_arrays_equal(st0,
                            SpikeTrain(numpy.arange(0, 101, dtype=float),
                                       t_start=0*pq.s,
                                       t_stop=101*pq.ms,
                                       units=pq.ms))
        st4 = segment.spiketrains[4]
        self.assertIsInstance(st4, SpikeTrain)
        assert_arrays_equal(st4,
                            SpikeTrain(numpy.arange(4, 105, dtype=float),
                                       t_start=0*pq.s,
                                       t_stop=105*pq.ms,
                                       units=pq.ms))
        # test annotations (stuff from file metadata)

    def test_read_spiketrain_using_eager(self):
        io = self.io_cls(self.test_file)
        st3 = io.read_spiketrain(lazy=False, channel_index=3)
        self.assertIsInstance(st3, SpikeTrain)
        assert_arrays_equal(st3,
                            SpikeTrain(numpy.arange(3, 104, dtype=float),
                                       t_start=0*pq.s,
                                       t_stop=104*pq.s,
                                       units=pq.ms))
        # should test annotations: 'channel_index', etc.

    def test_read_analogsignal_should_fail_with_spiketrain_file(self):
        io = self.io_cls(self.test_file)
        self.assertRaises(TypeError, io.read_analogsignal, channel_index=2)


class BaseTestPyNNNumpyIO(object):
    io_cls = PyNNNumpyIO
    file_extension = "npz"
    
    def write_test_file(self, variable='v', check=False):
        data, metadata = self.build_test_data(variable)
        metadata_array = numpy.array(sorted(metadata.items()))
        numpy.savez(self.test_file, data=data, metadata=metadata_array)
        if check:
            data1, metadata1 = read_test_file(self.test_file)
            assert metadata == metadata1, "%s != %s" % (metadata, metadata1)
            assert data.shape == data1.shape == (505, 2), "%s, %s, (505, 2)" % (data.shape, data1.shape)
            assert (data == data1).all()
            assert metadata["n"] == 505
    write_test_file.__test__ = False


class BaseTestPyNNTextIO(object):
    io_cls = PyNNTextIO
    file_extension = "txt"
    
    def write_test_file(self, variable='v', check=False):
        data, metadata = self.build_test_data(variable)
        with open(self.test_file, 'wb') as f:
            for item in sorted(metadata.items()):
                f.write(("# %s = %s\n" % item).encode('utf8'))
            numpy.savetxt(f, data)
        if check:
            raise NotImplementedError
    write_test_file.__test__ = False


class TestPyNNNumpyIO_Signals(BaseTestPyNNNumpyIO, BaseTestPyNNIO_Signals, unittest.TestCase):
    __test__ = True
    

class TestPyNNNumpyIO_Spikes(BaseTestPyNNNumpyIO, BaseTestPyNNIO_Spikes, unittest.TestCase):
    __test__ = True


class TestPyNNTextIO_Signals(BaseTestPyNNTextIO, BaseTestPyNNIO_Signals, unittest.TestCase):
    __test__ = True


class TestPyNNTextIO_Spikes(BaseTestPyNNTextIO, BaseTestPyNNIO_Spikes, unittest.TestCase):
    __test__ = True


if __name__ == '__main__':
    unittest.main()
