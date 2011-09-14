"""
Tests of the RecordingChannelGroup class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
    
from neo.core.recordingchannelgroup import RecordingChannelGroup

class TestRecordingChannelGroup(unittest.TestCase):
    def testInitDefaults(self):
        rcg = RecordingChannelGroup()
        self.assertEqual(rcg.name, None)
        self.assertEqual(rcg.file_origin, None)
        self.assertEqual(rcg.recordingchannels, [])
        self.assertEqual(rcg.analogsignalarrays, [])
        self.assertEqual(rcg.channel_names, [])
        self.assertEqual(rcg.channel_indexes, [])
    
    def testInit(self):
        rcg = RecordingChannelGroup(file_origin='temp.dat', channel_indexes=[1])
        self.assertEqual(rcg.file_origin, 'temp.dat')
        self.assertEqual(rcg.name, None)
        self.assertEqual(rcg.recordingchannels, [])
        self.assertEqual(rcg.analogsignalarrays, [])
        self.assertEqual(rcg.channel_names, [])
        self.assertEqual(rcg.channel_indexes, [1])        


if __name__ == '__main__':
    unittest.main()