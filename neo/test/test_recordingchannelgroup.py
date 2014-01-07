# -*- coding: utf-8 -*-
"""
Tests of the neo.core.recordingchannelgroup.RecordingChannelGroup class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core.recordingchannelgroup import RecordingChannelGroup
from neo.core.analogsignalarray import AnalogSignalArray
from neo.core.recordingchannel import RecordingChannel
from neo.core.unit import Unit
from neo.test.tools import assert_arrays_equal, assert_neo_object_is_compliant
from neo.io.tools import create_many_to_one_relationship


class TestRecordingChannelGroup(unittest.TestCase):
    def setUp(self):
        self.setup_unit()
        self.setup_analogsignalarrays()
        self.setup_recordingchannels()
        self.setup_recordingchannelgroups()

    def setup_recordingchannelgroups(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        self.rcg1 = RecordingChannelGroup(name='test', description='tester 1',
                                          file_origin='test.file',
                                          testarg1=1, **params)
        self.rcg2 = RecordingChannelGroup(name='test', description='tester 2',
                                          file_origin='test.file',
                                          testarg1=1, **params)
        self.rcg1.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        self.rcg2.annotate(testarg11=1.1, testarg10=[1, 2, 3])

        self.rcg1.units = self.units1
        self.rcg2.units = self.units2
        self.rcg1.recordingchannels = self.rchan1
        self.rcg2.recordingchannels = self.rchan2
        self.rcg1.analogsignalarrays = self.sigarr1
        self.rcg2.analogsignalarrays = self.sigarr2

        create_many_to_one_relationship(self.rcg1)
        create_many_to_one_relationship(self.rcg2)

    def setup_unit(self):
        unitname11 = 'unit 1 1'
        unitname12 = 'unit 1 2'
        unitname21 = 'unit 2 1'
        unitname22 = 'unit 2 2'

        self.unitnames1 = [unitname11, unitname12]
        self.unitnames2 = [unitname21, unitname22, unitname11]
        self.unitnames = [unitname11, unitname12, unitname21, unitname22]

        unit11 = Unit(name=unitname11, channel_indexes=np.array([1]))
        unit12 = Unit(name=unitname12, channel_indexes=np.array([2]))
        unit21 = Unit(name=unitname21, channel_indexes=np.array([1]))
        unit22 = Unit(name=unitname22, channel_indexes=np.array([2]))
        unit23 = Unit(name=unitname11, channel_indexes=np.array([1]))

        self.units1 = [unit11, unit12]
        self.units2 = [unit21, unit22, unit23]
        self.units = [unit11, unit12, unit21, unit22]

    def setup_recordingchannels(self):
        rchanname11 = 'chan 1 1'
        rchanname12 = 'chan 1 2'
        rchanname21 = 'chan 2 1'
        rchanname22 = 'chan 2 2'

        self.rchannames1 = [rchanname11, rchanname12]
        self.rchannames2 = [rchanname21, rchanname22, rchanname11]
        self.rchannames = [rchanname11, rchanname12, rchanname21, rchanname22]

        rchan11 = RecordingChannel(name=rchanname11)
        rchan12 = RecordingChannel(name=rchanname12)
        rchan21 = RecordingChannel(name=rchanname21)
        rchan22 = RecordingChannel(name=rchanname22)
        rchan23 = RecordingChannel(name=rchanname11)

        self.rchan1 = [rchan11, rchan12]
        self.rchan2 = [rchan21, rchan22, rchan23]
        self.rchan = [rchan11, rchan12, rchan21, rchan22]

    def setup_analogsignalarrays(self):
        sigarrname11 = 'analogsignalarray 1 1'
        sigarrname12 = 'analogsignalarray 1 2'
        sigarrname21 = 'analogsignalarray 2 1'
        sigarrname22 = 'analogsignalarray 2 2'

        sigarrdata11 = np.arange(0, 10).reshape(5, 2) * pq.mV
        sigarrdata12 = np.arange(10, 20).reshape(5, 2) * pq.mV
        sigarrdata21 = np.arange(20, 30).reshape(5, 2) * pq.V
        sigarrdata22 = np.arange(30, 40).reshape(5, 2) * pq.V
        sigarrdata112 = np.hstack([sigarrdata11, sigarrdata11]) * pq.mV

        self.sigarrnames1 = [sigarrname11, sigarrname12]
        self.sigarrnames2 = [sigarrname21, sigarrname22, sigarrname11]
        self.sigarrnames = [sigarrname11, sigarrname12,
                            sigarrname21, sigarrname22]

        sigarr11 = AnalogSignalArray(sigarrdata11, name=sigarrname11,
                                     sampling_rate=1*pq.Hz,
                                     channel_index=np.array([1]))
        sigarr12 = AnalogSignalArray(sigarrdata12, name=sigarrname12,
                                     sampling_rate=1*pq.Hz,
                                     channel_index=np.array([2]))
        sigarr21 = AnalogSignalArray(sigarrdata21, name=sigarrname21,
                                     sampling_rate=1*pq.Hz,
                                     channel_index=np.array([1]))
        sigarr22 = AnalogSignalArray(sigarrdata22, name=sigarrname22,
                                     sampling_rate=1*pq.Hz,
                                     channel_index=np.array([2]))
        sigarr23 = AnalogSignalArray(sigarrdata11, name=sigarrname11,
                                     sampling_rate=1*pq.Hz,
                                     channel_index=np.array([1]))
        sigarr112 = AnalogSignalArray(sigarrdata112, name=sigarrname11,
                                      sampling_rate=1*pq.Hz,
                                      channel_index=np.array([1]))

        self.sigarr1 = [sigarr11, sigarr12]
        self.sigarr2 = [sigarr21, sigarr22, sigarr23]
        self.sigarr = [sigarr112, sigarr12, sigarr21, sigarr22]

    def test__recordingchannelgroup__init_defaults(self):
        rcg = RecordingChannelGroup()
        assert_neo_object_is_compliant(rcg)
        self.assertEqual(rcg.name, None)
        self.assertEqual(rcg.file_origin, None)
        self.assertEqual(rcg.recordingchannels, [])
        self.assertEqual(rcg.analogsignalarrays, [])
        assert_arrays_equal(rcg.channel_names, np.array([], dtype='S'))
        assert_arrays_equal(rcg.channel_indexes, np.array([]))

    def test_recordingchannelgroup__init(self):
        rcg = RecordingChannelGroup(file_origin='temp.dat',
                                    channel_indexes=np.array([1]))
        assert_neo_object_is_compliant(rcg)
        self.assertEqual(rcg.file_origin, 'temp.dat')
        self.assertEqual(rcg.name, None)
        self.assertEqual(rcg.recordingchannels, [])
        self.assertEqual(rcg.analogsignalarrays, [])
        assert_arrays_equal(rcg.channel_names, np.array([], dtype='S'))
        assert_arrays_equal(rcg.channel_indexes, np.array([1]))

    def test_recordingchannelgroup__compliance(self):
        assert_neo_object_is_compliant(self.rcg1)
        assert_neo_object_is_compliant(self.rcg2)

        self.assertEqual(self.rcg1.name, 'test')
        self.assertEqual(self.rcg2.name, 'test')

        self.assertEqual(self.rcg1.description, 'tester 1')
        self.assertEqual(self.rcg2.description, 'tester 2')

        self.assertEqual(self.rcg1.file_origin, 'test.file')
        self.assertEqual(self.rcg2.file_origin, 'test.file')

        self.assertEqual(self.rcg1.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(self.rcg2.annotations['testarg10'], [1, 2, 3])

        self.assertEqual(self.rcg1.annotations['testarg1'], 1.1)
        self.assertEqual(self.rcg2.annotations['testarg1'], 1)
        self.assertEqual(self.rcg2.annotations['testarg11'], 1.1)

        self.assertEqual(self.rcg1.annotations['testarg2'], 'yes')
        self.assertEqual(self.rcg2.annotations['testarg2'], 'yes')

        self.assertTrue(self.rcg1.annotations['testarg3'])
        self.assertTrue(self.rcg2.annotations['testarg3'])

        self.assertTrue(hasattr(self.rcg1, 'units'))
        self.assertTrue(hasattr(self.rcg2, 'units'))

        self.assertEqual(len(self.rcg1.units), 2)
        self.assertEqual(len(self.rcg2.units), 3)

        self.assertEqual(self.rcg1.units, self.units1)
        self.assertEqual(self.rcg2.units, self.units2)

        self.assertTrue(hasattr(self.rcg1, 'recordingchannels'))
        self.assertTrue(hasattr(self.rcg2, 'recordingchannels'))

        self.assertEqual(len(self.rcg1.recordingchannels), 2)
        self.assertEqual(len(self.rcg2.recordingchannels), 3)

        for res, targ in zip(self.rcg1.recordingchannels, self.rchan1):
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.rcg2.recordingchannels, self.rchan2):
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.rcg1, 'analogsignalarrays'))
        self.assertTrue(hasattr(self.rcg2, 'analogsignalarrays'))

        self.assertEqual(len(self.rcg1.analogsignalarrays), 2)
        self.assertEqual(len(self.rcg2.analogsignalarrays), 3)

        for res, targ in zip(self.rcg1.analogsignalarrays, self.sigarr1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.rcg2.analogsignalarrays, self.sigarr2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

    def test_recordingchannelgroup__merge(self):
        self.rcg1.merge(self.rcg2)

        chanres1 = [chan.name for chan in self.rcg1.recordingchannels]
        chanres2 = [chan.name for chan in self.rcg2.recordingchannels]

        unitres1 = [unit.name for unit in self.rcg1.units]
        unitres2 = [unit.name for unit in self.rcg2.units]

        sigarrres1 = [sigarr.name for sigarr in self.rcg1.analogsignalarrays]
        sigarrres2 = [sigarr.name for sigarr in self.rcg2.analogsignalarrays]

        self.assertEqual(chanres1, self.rchannames)
        self.assertEqual(chanres2, self.rchannames2)

        self.assertEqual(unitres1, self.unitnames)
        self.assertEqual(unitres2, self.unitnames2)

        self.assertEqual(sigarrres1, self.sigarrnames)
        self.assertEqual(sigarrres2, self.sigarrnames2)

        for res, targ in zip(self.rcg1.analogsignalarrays, self.sigarr):
            assert_arrays_equal(res, targ)

        for res, targ in zip(self.rcg2.analogsignalarrays,  self.sigarr2):
            assert_arrays_equal(res, targ)


if __name__ == '__main__':
    unittest.main()
