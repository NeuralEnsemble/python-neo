# add performance testing!!

import numpy as np
import quantities as pq
try:
    import unittest2 as unittest
except ImportError:
    import unittest
import datetime
import os

from neo.core import *
try:
    from neo.io.hdf5io import IOManager
    have_hdf5 = True
except ImportError:
    have_hdf5 = False

from neo.test.io.common_io_test import BaseTestIO

#===============================================================================

def checks():
    block = Block()
    block._segments = []
    for i in range(4):
        block._segments.append(Segment())
    segment = block._segments[0]
    segment._events = []
    for i in range(5):
        segment._events.append(Event(time=120 * pq.millisecond, \
            label="Black Label")) # hehe
    segment._eventarrays = []
    for i in range(15):
        segment._eventarrays.append(EventArray(times=np.random.rand(20) * pq.second, \
            labels=np.array("", dtype="a100")))
    segment._epochs = []
    for i in range(5):
        segment._epochs.append(Epoch(time=120 * pq.millisecond, \
            duration = 845.0 * pq.millisecond, label="Red Label")) # hehe
    segment._epocharrays = []
    for i in range(15):
        segment._epocharrays.append(EpochArray(times=np.random.rand(20) \
            * pq.second, durations=np.random.rand(20) \
            * pq.second, labels=np.array("", dtype="a100")))
    segment._spiketrains = []
    for i in range(30):
        segment._spiketrains.append(SpikeTrain(t_start=-200.0 * pq.millisecond, \
            t_stop=200.0 * pq.millisecond, times=np.random.rand(20) * pq.millisecond, \
            waveforms=np.random.rand(20, 2, 100) * pq.millisecond))
    segment._analogsignals = []
    for i in range(300):
        segment._analogsignals.append(AnalogSignal(name="AS-TEST" + str(i), \
            t_start=-200.0 * pq.millisecond, sampling_rate=10000.0 * pq.hertz, \
                signal=np.random.rand(500) * pq.millivolt))
    segment._irsaanalogsignals = []
    for i in range(50):
        segment._irsaanalogsignals.append(IrregularlySampledSignal(name="IRSA-TEST" + str(i), \
            t_start=260.0 * pq.millisecond, channel_name="2", \
            signal=np.random.rand(500) * pq.millivolt, \
            times=np.random.rand(20) * pq.millisecond))
    segment._analogsignalarrays = []
    #for i in range(5):
    #    segment._analogsignalarrays.append(AnalogSignalArray(t_start=260.0 \
    #        * pq.millisecond, channel_names="Don't know..", \
    #        signal=np.random.rand(10, 500) * pq.millivolt))
    segment._spikes = []
    for i in range(97):
        segment._spikes.append(Spike(time=-260.0 * pq.millisecond, \
            sampling_rate=10000.0 * pq.hertz, left_sweep=-108.0 * pq.millisecond, \
            waveform=np.random.rand(10, 500) * pq.millivolt))
    block._segments.append(segment)
    
    recch = RecordingChannel(name="Test rec Group", index=i)
    recch._units = []
    for i in range(3):
        recch._units.append(Unit(name="Neuron #" + str(i)))
    recchgrp = RecordingChannelGroup(name="Test rec Group")
    recchgrp._recordingchannels = []
    recchgrp._recordingchannels.append(recch)
    for i in range(7):
        recchgrp._recordingchannels.append(RecordingChannel(name="Test rec Group", \
            index=i))
    block._recordingchannelgroups = []
    block._recordingchannelgroups.append(recchgrp)
    return block


class hdf5ioTest(unittest.TestCase, BaseTestIO):
    """
    Tests for the hdf5 library.
    """
    ioclass = IOManager
    
    @unittest.skipUnless(have_hdf5, "requires PyTables")
    def setUp(self):
        self.test_file = "test.h5"
    
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_create(self):
        """
        Create test file with signals, segments, blocks etc.
        """
        iom = IOManager(filename=self.test_file)
        # creating a structure
        block = checks()
        # saving & testing
        iom.save(block)
        iom.close()
        iom.connect(filename=self.test_file)
        self.assertEqual(len(iom._data.listNodes("/")), 1)
        self.assertEqual(len(iom._data.listNodes("/block_0/_segments")), 4)
        self.assertEqual(len(iom._data.listNodes("/block_0/_recordingchannelgroups")), 1)
        self.assertEqual(len(iom._data.listNodes("/block_0/_recordingchannelgroups/recordingchannelgroup_0/_recordingchannels")), 8)
        self.assertEqual(len(iom._data.listNodes("/block_0/_recordingchannelgroups/recordingchannelgroup_0/_recordingchannels/recordingchannel_0/_units")), 3)
        self.assertEqual(len(iom._data.listNodes("/block_0/_segments/segment_0/_events")), 5)
        self.assertEqual(len(iom._data.listNodes("/block_0/_segments/segment_0/_eventarrays")), 15)
        self.assertEqual(len(iom._data.listNodes("/block_0/_segments/segment_0/_epochs")), 5)
        self.assertEqual(len(iom._data.listNodes("/block_0/_segments/segment_0/_epocharrays")), 15)
        self.assertEqual(len(iom._data.listNodes("/block_0/_segments/segment_0/_spiketrains")), 30)
        self.assertEqual(len(iom._data.listNodes("/block_0/_segments/segment_0/_analogsignals")), 300)
        self.assertEqual(len(iom._data.listNodes("/block_0/_segments/segment_0/_irsaanalogsignals")), 50)
        #self.assertEqual(len(iom._data.listNodes("/block_0/_segments/segment_0/_analogsignalarrays")), 5)
        self.assertEqual(len(iom._data.listNodes("/block_0/_segments/segment_0/_spikes")), 97)


    #def test_relations(self):
    #    pass

    #def test_property_change(self):
    #    pass

    #def test_get_objects(self):
    #    pass
        
    #def test_data_change(self):
    #    pass

if __name__ == '__main__':
    unittest.main()



