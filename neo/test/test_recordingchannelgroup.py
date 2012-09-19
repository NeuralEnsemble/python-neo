"""
Tests of the RecordingChannelGroup class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.core.recordingchannelgroup import RecordingChannelGroup
import numpy as np

from neo.test.tools import assert_arrays_equal

class TestRecordingChannelGroup(unittest.TestCase):
    def testInitDefaults(self):
        rcg = RecordingChannelGroup()
        self.assertEqual(rcg.name, None)
        self.assertEqual(rcg.file_origin, None)
        self.assertEqual(rcg.recordingchannels, [])
        self.assertEqual(rcg.analogsignalarrays, [])
        assert_arrays_equal(rcg.channel_names, np.array([]))
        assert_arrays_equal(rcg.channel_indexes, np.array([]))

    def testInit(self):
        rcg = RecordingChannelGroup(file_origin='temp.dat', channel_indexes=np.array([1]))
        self.assertEqual(rcg.file_origin, 'temp.dat')
        self.assertEqual(rcg.name, None)
        self.assertEqual(rcg.recordingchannels, [])
        self.assertEqual(rcg.analogsignalarrays, [])
        assert_arrays_equal(rcg.channel_names, np.array([]))
        assert_arrays_equal(rcg.channel_indexes, np.array([1]))


if __name__ == '__main__':
    unittest.main()
