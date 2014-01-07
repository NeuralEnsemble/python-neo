# -*- coding: utf-8 -*-
"""
Tests of the neo.core.recordingchannel.RecordingChannel class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core.recordingchannel import RecordingChannel
from neo.core.analogsignal import AnalogSignal
from neo.core.irregularlysampledsignal import IrregularlySampledSignal
from neo.test.tools import assert_neo_object_is_compliant, assert_arrays_equal
from neo.io.tools import create_many_to_one_relationship


class TestRecordingChannel(unittest.TestCase):
    def setUp(self):
        self.setup_analogsignals()
        self.setup_irregularlysampledsignals()
        self.setup_recordingchannels()

    def setup_recordingchannels(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        self.rchan1 = RecordingChannel(index=10,
                                       coordinate=[1.1, 1.5, 1.7]*pq.mm,
                                       name='test', description='tester 1',
                                       file_origin='test.file',
                                       testarg1=1, **params)
        self.rchan2 = RecordingChannel(index=100,
                                       coordinate=[11., 15., 17.]*pq.mm,
                                       name='test', description='tester 2',
                                       file_origin='test.file',
                                       testarg1=1, **params)
        self.rchan1.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        self.rchan2.annotate(testarg11=1.1, testarg10=[1, 2, 3])

        self.rchan1.analogsignals = self.sig1
        self.rchan2.analogsignals = self.sig2

        self.rchan1.irregularlysampledsignals = self.irsig1
        self.rchan2.irregularlysampledsignals = self.irsig2

        create_many_to_one_relationship(self.rchan1)
        create_many_to_one_relationship(self.rchan2)

    def setup_analogsignals(self):
        signame11 = 'analogsignal 1 1'
        signame12 = 'analogsignal 1 2'
        signame21 = 'analogsignal 2 1'
        signame22 = 'analogsignal 2 2'

        sigdata11 = np.arange(0, 10) * pq.mV
        sigdata12 = np.arange(10, 20) * pq.mV
        sigdata21 = np.arange(20, 30) * pq.V
        sigdata22 = np.arange(30, 40) * pq.V

        self.signames1 = [signame11, signame12]
        self.signames2 = [signame21, signame22]
        self.signames = [signame11, signame12, signame21, signame22]

        sig11 = AnalogSignal(sigdata11, name=signame11, sampling_rate=1*pq.Hz)
        sig12 = AnalogSignal(sigdata12, name=signame12, sampling_rate=1*pq.Hz)
        sig21 = AnalogSignal(sigdata21, name=signame21, sampling_rate=1*pq.Hz)
        sig22 = AnalogSignal(sigdata22, name=signame22, sampling_rate=1*pq.Hz)

        self.sig1 = [sig11, sig12]
        self.sig2 = [sig21, sig22]
        self.sig = [sig11, sig12, sig21, sig22]

    def setup_irregularlysampledsignals(self):
        irsigname11 = 'irregularsignal 1 1'
        irsigname12 = 'irregularsignal 1 2'
        irsigname21 = 'irregularsignal 2 1'
        irsigname22 = 'irregularsignal 2 2'

        irsigdata11 = np.arange(0, 10) * pq.mA
        irsigdata12 = np.arange(10, 20) * pq.mA
        irsigdata21 = np.arange(20, 30) * pq.A
        irsigdata22 = np.arange(30, 40) * pq.A

        irsigtimes11 = np.arange(0, 10) * pq.ms
        irsigtimes12 = np.arange(10, 20) * pq.ms
        irsigtimes21 = np.arange(20, 30) * pq.s
        irsigtimes22 = np.arange(30, 40) * pq.s

        self.irsignames1 = [irsigname11, irsigname12]
        self.irsignames2 = [irsigname21, irsigname22]
        self.irsignames = [irsigname11, irsigname12, irsigname21, irsigname22]

        irsig11 = IrregularlySampledSignal(irsigtimes11, irsigdata11,
                                           name=irsigname11)
        irsig12 = IrregularlySampledSignal(irsigtimes12, irsigdata12,
                                           name=irsigname12)
        irsig21 = IrregularlySampledSignal(irsigtimes21, irsigdata21,
                                           name=irsigname21)
        irsig22 = IrregularlySampledSignal(irsigtimes22, irsigdata22,
                                           name=irsigname22)

        self.irsig1 = [irsig11, irsig12]
        self.irsig2 = [irsig21, irsig22]
        self.irsig = [irsig11, irsig12, irsig21, irsig22]

    def test_recordingchannel_creation(self):
        assert_neo_object_is_compliant(self.rchan1)
        assert_neo_object_is_compliant(self.rchan2)

        self.assertEqual(self.rchan1.index, 10)
        self.assertEqual(self.rchan2.index, 100)

        assert_arrays_equal(self.rchan1.coordinate, [1.1, 1.5, 1.7]*pq.mm)
        assert_arrays_equal(self.rchan2.coordinate, [11., 15., 17.]*pq.mm)

        self.assertEqual(self.rchan1.name, 'test')
        self.assertEqual(self.rchan2.name, 'test')

        self.assertEqual(self.rchan1.description, 'tester 1')
        self.assertEqual(self.rchan2.description, 'tester 2')

        self.assertEqual(self.rchan1.file_origin, 'test.file')
        self.assertEqual(self.rchan2.file_origin, 'test.file')

        self.assertEqual(self.rchan1.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(self.rchan2.annotations['testarg10'], [1, 2, 3])

        self.assertEqual(self.rchan1.annotations['testarg1'], 1.1)
        self.assertEqual(self.rchan2.annotations['testarg1'], 1)
        self.assertEqual(self.rchan2.annotations['testarg11'], 1.1)

        self.assertEqual(self.rchan1.annotations['testarg2'], 'yes')
        self.assertEqual(self.rchan2.annotations['testarg2'], 'yes')

        self.assertTrue(self.rchan1.annotations['testarg3'])
        self.assertTrue(self.rchan2.annotations['testarg3'])

        self.assertTrue(hasattr(self.rchan1, 'analogsignals'))
        self.assertTrue(hasattr(self.rchan2, 'analogsignals'))

        self.assertEqual(len(self.rchan1.analogsignals), 2)
        self.assertEqual(len(self.rchan2.analogsignals), 2)

        for res, targ in zip(self.rchan1.analogsignals, self.sig1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.rchan2.analogsignals, self.sig2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.rchan1, 'irregularlysampledsignals'))
        self.assertTrue(hasattr(self.rchan2, 'irregularlysampledsignals'))

        self.assertEqual(len(self.rchan1.irregularlysampledsignals), 2)
        self.assertEqual(len(self.rchan2.irregularlysampledsignals), 2)

        for res, targ in zip(self.rchan1.irregularlysampledsignals,
                             self.irsig1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.rchan2.irregularlysampledsignals,
                             self.irsig2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.rchan1, 'recordingchannelgroups'))
        self.assertTrue(hasattr(self.rchan2, 'recordingchannelgroups'))

    def test_recordingchannel_merge(self):
        self.rchan1.merge(self.rchan2)

        sigres1 = [sig.name for sig in self.rchan1.analogsignals]
        sigres2 = [sig.name for sig in self.rchan2.analogsignals]

        irsigres1 = [sig.name for sig in self.rchan1.irregularlysampledsignals]
        irsigres2 = [sig.name for sig in self.rchan2.irregularlysampledsignals]

        self.assertEqual(sigres1, self.signames)
        self.assertEqual(sigres2, self.signames2)

        self.assertEqual(irsigres1, self.irsignames)
        self.assertEqual(irsigres2, self.irsignames2)

        for res, targ in zip(self.rchan1.analogsignals, self.sig):
            assert_arrays_equal(res, targ)

        for res, targ in zip(self.rchan2.analogsignals, self.sig2):
            assert_arrays_equal(res, targ)

        for res, targ in zip(self.rchan1.irregularlysampledsignals,
                             self.irsig):
            assert_arrays_equal(res, targ)

        for res, targ in zip(self.rchan2.irregularlysampledsignals,
                             self.irsig2):
            assert_arrays_equal(res, targ)


if __name__ == "__main__":
    unittest.main()
