# -*- coding: utf-8 -*-
"""
Tests of neo.io.blackrockio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import os
import sys
import re
import warnings

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo import NeuralynxIO, AnalogSignal, SpikeTrain, Event
from neo.test.iotest.common_io_test import BaseTestIO
from neo.core import Segment


class CommonTests(BaseTestIO):
    ioclass = NeuralynxIO
    files_to_test = []
    files_to_download = [
        'Cheetah_v5.5.1/original_data/CheetahLogFile.txt',
        'Cheetah_v5.5.1/original_data/CheetahLostADRecords.txt',
        'Cheetah_v5.5.1/original_data/Events.nev',
        'Cheetah_v5.5.1/original_data/STet3a.nse',
        'Cheetah_v5.5.1/original_data/STet3b.nse',
        'Cheetah_v5.5.1/original_data/Tet3a.ncs',
        'Cheetah_v5.5.1/original_data/Tet3b.ncs',
        'Cheetah_v5.5.1/plain_data/STet3a.txt',
        'Cheetah_v5.5.1/plain_data/STet3b.txt',
        'Cheetah_v5.5.1/plain_data/Tet3a.txt',
        'Cheetah_v5.5.1/plain_data/Tet3b.txt',
        'Cheetah_v5.5.1/plain_data/Events.txt',
        'Cheetah_v5.5.1/README.txt',
        'Cheetah_v5.7.4/original_data/CSC1.ncs',
        'Cheetah_v5.7.4/original_data/CSC2.ncs',
        'Cheetah_v5.7.4/original_data/CSC3.ncs',
        'Cheetah_v5.7.4/original_data/CSC4.ncs',
        'Cheetah_v5.7.4/original_data/CSC5.ncs',
        'Cheetah_v5.7.4/original_data/Events.nev',
        'Cheetah_v5.7.4/plain_data/CSC1.txt',
        'Cheetah_v5.7.4/plain_data/CSC2.txt',
        'Cheetah_v5.7.4/plain_data/CSC3.txt',
        'Cheetah_v5.7.4/plain_data/CSC4.txt',
        'Cheetah_v5.7.4/plain_data/CSC5.txt',
        'Cheetah_v5.7.4/plain_data/Events.txt',
        'Cheetah_v5.7.4/README.txt']

    def setUp(self):
        super(CommonTests, self).setUp()
        data_dir = os.path.join(self.local_test_dir,
                                'Cheetah_v{}'.format(self.cheetah_version))

        self.sn = os.path.join(data_dir, 'original_data')
        self.pd = os.path.join(data_dir, 'plain_data')
        if not os.path.exists(self.sn):
            raise unittest.SkipTest('data file does not exist:' + self.sn)


class TestCheetah_v551(CommonTests, unittest.TestCase):
    cheetah_version = '5.5.1'

    def test_read_block(self):
        """Read data in a certain time range into one block"""
        t_start, t_stop = 3 * pq.s, 4 * pq.s

        nio = NeuralynxIO(self.sn, use_cache='never')
        block = nio.read_block(t_starts=[t_start], t_stops=[t_stop])
        self.assertEqual(len(nio.parameters_ncs), 2)
        self.assertTrue(
                {'event_id': 11, 'name': 'Starting Recording', 'nttl': 0} in
                nio.parameters_nev['Events.nev']['event_types'])

        # Everything put in one segment
        self.assertEqual(len(block.segments), 1)
        seg = block.segments[0]
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape[-1], 2)

        self.assertEqual(seg.analogsignals[0].sampling_rate.units,
                         pq.CompoundUnit('32*kHz'))
        self.assertEqual(seg.analogsignals[0].t_start, t_start)
        self.assertEqual(seg.analogsignals[0].t_stop, t_stop)
        self.assertEqual(len(seg.spiketrains), 2)

        # Testing different parameter combinations
        block = nio.read_block(lazy=True)
        self.assertEqual(len(block.segments[0].analogsignals[0]), 0)
        self.assertEqual(len(block.segments[0].spiketrains[0]), 0)

        block = nio.read_block(cascade=False)
        self.assertEqual(len(block.segments), 0)

        block = nio.read_block(electrode_list=[0])
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(len(block.channel_indexes[-1].units), 1)

        block = nio.read_block(t_starts=None, t_stops=None, events=True,
                               waveforms=True)
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(len(block.segments[0].spiketrains), 2)
        self.assertEqual(len(block.segments[0].spiketrains[0].waveforms),
                         len(block.segments[0].spiketrains[0]))
        self.assertGreater(len(block.segments[0].events), 0)
        self.assertEqual(len(block.channel_indexes[-1].units), 2)

        block = nio.read_block(t_starts=[t_start], t_stops=[t_stop],
                               unit_list=[0], electrode_list=[0])
        self.assertEqual(len(block.channel_indexes[-1].units), 1)

        block = nio.read_block(t_starts=[t_start], t_stops=[t_stop],
                               unit_list=False)
        self.assertEqual(len(block.channel_indexes[-1].units), 0)

    def test_read_segment(self):
        """Read data in a certain time range into one block"""

        nio = NeuralynxIO(self.sn, use_cache='never')
        seg = nio.read_segment(t_start=None, t_stop=None)

        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape[-1], 2)

        self.assertEqual(seg.analogsignals[0].sampling_rate.units,
                         pq.CompoundUnit('32*kHz'))

        self.assertEqual(len(seg.spiketrains), 2)

        # Testing different parameter combinations
        seg = nio.read_segment(lazy=True)
        self.assertEqual(len(seg.analogsignals[0]), 0)
        self.assertEqual(len(seg.spiketrains[0]), 0)

        seg = nio.read_segment(cascade=False)
        self.assertEqual(len(seg.analogsignals), 0)
        self.assertEqual(len(seg.spiketrains), 0)

        seg = nio.read_segment(electrode_list=[0])
        self.assertEqual(len(seg.analogsignals), 1)

        seg = nio.read_segment(t_start=None, t_stop=None, events=True,
                               waveforms=True)
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(len(seg.spiketrains), 2)
        self.assertTrue(len(seg.spiketrains[0].waveforms) > 0)
        self.assertTrue(len(seg.events) > 0)

    def test_read_ncs_data(self):
        t_start, t_stop = 0, 500 * 512  # in samples

        nio = NeuralynxIO(self.sn, use_cache='never')
        seg = Segment('testsegment')

        for el_id, el_dict in nio.parameters_ncs.iteritems():
            filepath = nio.parameters_ncs[el_id]['recording_file_name']
            filename = filepath.split('/')[-1].split('\\')[-1].split('.')[0]
            nio.read_ncs(filename, seg, t_start=t_start, t_stop=t_stop)
            anasig = seg.filter({'electrode_id': el_id},
                                objects=AnalogSignal)[0]

            target_data = np.zeros((16679, 512))
            with open(self.pd + '/%s.txt' % filename) as datafile:
                for i, line in enumerate(datafile):
                    line = line.strip('\xef\xbb\xbf')
                    entries = line.split()
                    target_data[i, :] = np.asarray(entries[4:])

            target_data = target_data.reshape((-1, 1)) * el_dict['ADBitVolts']

            np.testing.assert_array_equal(target_data[:len(anasig)],
                                          anasig.magnitude)

    def test_read_nse_data(self):
        t_start, t_stop = None, None  # in samples

        nio = NeuralynxIO(self.sn, use_cache='never')
        seg = Segment('testsegment')

        for el_id, el_dict in nio.parameters_nse.iteritems():
            filepath = nio.parameters_nse[el_id]['recording_file_name']
            filename = filepath.split('/')[-1].split('\\')[-1].split('.')[0]
            nio.read_nse(filename, seg, t_start=t_start, t_stop=t_stop,
                         waveforms=True)
            spiketrain = seg.filter({'electrode_id': el_id},
                                    objects=SpikeTrain)[0]

            # target_data = np.zeros((500, 32))
            # timestamps = np.zeros(500)
            entries = []
            with open(self.pd + '/%s.txt' % filename) as datafile:
                for i, line in enumerate(datafile):
                    line = line.strip('\xef\xbb\xbf')
                    entries.append(line.split())
            entries = np.asarray(entries, dtype=float)
            target_data = entries[:-1, 11:]
            timestamps = entries[:-1, 0]

            timestamps = (timestamps * pq.microsecond -
                          nio.parameters_global['t_start'])

            np.testing.assert_array_equal(timestamps.magnitude,
                                          spiketrain.magnitude)
            np.testing.assert_array_equal(target_data,
                                          spiketrain.waveforms)

    def test_read_nev_data(self):
        t_start, t_stop = 0 * pq.s, 1000 * pq.s

        nio = NeuralynxIO(self.sn, use_cache='never')
        seg = Segment('testsegment')

        filename = 'Events'
        nio.read_nev(filename + '.nev', seg, t_start=t_start, t_stop=t_stop)

        timestamps = []
        nttls = []
        names = []
        event_ids = []

        with open(self.pd + '/%s.txt' % filename) as datafile:
            for i, line in enumerate(datafile):
                line = line.strip('\xef\xbb\xbf')
                entries = line.split('\t')
                nttls.append(int(entries[5]))
                timestamps.append(int(entries[3]))
                names.append(entries[10].rstrip('\r\n'))
                event_ids.append(int(entries[4]))

        timestamps = (np.array(timestamps) * pq.microsecond -
                      nio.parameters_global['t_start'])
        # masking only requested spikes
        mask = np.where(timestamps < t_stop)[0]

        # return if no event fits criteria
        if len(mask) == 0:
            return
        timestamps = timestamps[mask]
        nttls = np.asarray(nttls)[mask]
        names = np.asarray(names)[mask]
        event_ids = np.asarray(event_ids)[mask]

        for i in range(len(timestamps)):
            events = seg.filter({'nttl': nttls[i]}, objects=Event)
            events = [e for e in events
                      if (e.annotations['marker_id'] == event_ids[i] and
                          e.labels == names[i])]
            self.assertTrue(len(events) == 1)
            self.assertTrue(timestamps[i] in events[0].times)

    def test_read_ntt_data(self):
        pass

        # TODO: Implement test_read_ntt_data once ntt files are available


class TestCheetah_v574(TestCheetah_v551, CommonTests, unittest.TestCase):
    cheetah_version = '5.7.4'

    def test_read_block(self):
        """Read data in a certain time range into one block"""
        t_start, t_stop = 3 * pq.s, 4 * pq.s

        nio = NeuralynxIO(self.sn, use_cache='never')
        block = nio.read_block(t_starts=[t_start], t_stops=[t_stop])
        self.assertEqual(len(nio.parameters_ncs), 5)
        self.assertTrue(
                {'event_id': 19, 'name': 'Starting Recording', 'nttl': 0} in
                nio.parameters_nev['Events.nev']['event_types'])
        self.assertTrue(
                {'event_id': 19, 'name': 'Stopping Recording', 'nttl': 0} in
                nio.parameters_nev['Events.nev']['event_types'])

        # Everything put in one segment
        self.assertEqual(len(block.segments), 1)
        seg = block.segments[0]
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape[-1], 5)

        self.assertEqual(seg.analogsignals[0].sampling_rate.units,
                         pq.CompoundUnit('32*kHz'))
        self.assertAlmostEqual(seg.analogsignals[0].t_start, t_start, places=4)
        self.assertAlmostEqual(seg.analogsignals[0].t_stop, t_stop, places=4)
        self.assertEqual(len(seg.spiketrains), 0)  # no nse files available

        # Testing different parameter combinations
        block = nio.read_block(lazy=True)
        self.assertEqual(len(block.segments[0].analogsignals[0]), 0)

        block = nio.read_block(cascade=False)
        self.assertEqual(len(block.segments), 0)

        block = nio.read_block(electrode_list=[0])
        self.assertEqual(len(block.segments[0].analogsignals), 1)

        block = nio.read_block(t_starts=None, t_stops=None, events=True,
                               waveforms=True)
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(len(block.segments[0].spiketrains), 0)
        self.assertGreater(len(block.segments[0].events), 0)
        self.assertEqual(len(block.channel_indexes), 5)

    def test_read_segment(self):
        """Read data in a certain time range into one block"""

        nio = NeuralynxIO(self.sn, use_cache='never')
        seg = nio.read_segment(t_start=None, t_stop=None)

        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape[-1], 5)

        self.assertEqual(seg.analogsignals[0].sampling_rate.units,
                         pq.CompoundUnit('32*kHz'))

        self.assertEqual(len(seg.spiketrains), 0)

        # Testing different parameter combinations
        seg = nio.read_segment(lazy=True)
        self.assertEqual(len(seg.analogsignals[0]), 0)
        self.assertEqual(len(seg.spiketrains), 0)

        seg = nio.read_segment(cascade=False)
        self.assertEqual(len(seg.analogsignals), 0)
        self.assertEqual(len(seg.spiketrains), 0)

        seg = nio.read_segment(electrode_list=[0])
        self.assertEqual(len(seg.analogsignals), 1)

        seg = nio.read_segment(t_start=None, t_stop=None, events=True,
                               waveforms=True)
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(len(seg.spiketrains), 0)
        self.assertTrue(len(seg.events) > 0)


class TestGaps(CommonTests, unittest.TestCase):
    cheetah_version = '5.5.1'

    def test_gap_handling(self):
        nio = NeuralynxIO(self.sn, use_cache='never')

        block = nio.read_block(t_starts=None, t_stops=None)

        # known gap values
        n_gaps = 1

        self.assertEqual(len(block.segments), n_gaps + 1)
        # one channel index for analogsignals for each of the 3 segments and
        # one for spiketrains
        self.assertEqual(len(block.channel_indexes), len(block.segments) + 1)
        self.assertEqual(len(block.channel_indexes[-1].units), 2)
        for unit in block.channel_indexes[-1].units:
            self.assertEqual(len(unit.spiketrains), n_gaps + 1)

        anasig_channels = [i for i in block.channel_indexes
                           if 'analogsignal' in i.name]
        self.assertEqual(len(anasig_channels), n_gaps + 1)

    def test_gap_warning(self):
        nio = NeuralynxIO(self.sn, use_cache='never')

        with reset_warning_registry():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                nio.read_block(t_starts=None, t_stops=None)

                self.assertGreater(len(w), 0)
                self.assertTrue(issubclass(w[0].category, UserWarning))
                self.assertEqual('Substituted t_starts and t_stops in order to'
                                 ' skip gap in recording session.',
                                 str(w[0].message))

    def test_analogsignal_shortening_warning(self):
        nio = NeuralynxIO(self.sn, use_cache='never')

        with reset_warning_registry():
            with warnings.catch_warnings(record=True) as w:
                seg = Segment('testsegment')
                nio.read_ncs(os.path.join(self.sn, 'Tet3a.ncs'), seg)

                self.assertGreater(len(w), 0)
                self.assertTrue(issubclass(w[0].category, UserWarning))
                self.assertTrue('Analogsignalarray was shortened due to gap in'
                                ' recorded data  of file'
                                in str(w[0].message))


# This class is copied from
# 'http://bugs.python.org/file40031/reset_warning_registry.py' by Eli Collins
# and is related to http://bugs.python.org/issue21724 Python<3.4
class reset_warning_registry(object):
    """
    context manager which archives & clears warning registry for duration of
    context.

    :param pattern:
          optional regex pattern, causes manager to only reset modules whose
          names match this pattern. defaults to ``".*"``.
    """

    #: regexp for filtering which modules are reset
    _pattern = None

    #: dict mapping module name -> old registry contents
    _backup = None

    def __init__(self, pattern=None):
        self._pattern = re.compile(pattern or ".*")

    def __enter__(self):
        # archive and clear the __warningregistry__ key for all modules
        # that match the 'reset' pattern.
        pattern = self._pattern
        backup = self._backup = {}
        for name, mod in list(sys.modules.items()):
            if pattern.match(name):
                reg = getattr(mod, "__warningregistry__", None)
                if reg:
                    backup[name] = reg.copy()
                    reg.clear()
        return self

    def __exit__(self, *exc_info):
        # restore warning registry from backup
        modules = sys.modules
        backup = self._backup
        for name, content in backup.items():
            mod = modules.get(name)
            if mod is None:
                continue
            reg = getattr(mod, "__warningregistry__", None)
            if reg is None:
                setattr(mod, "__warningregistry__", content)
            else:
                reg.clear()
                reg.update(content)

        # clear all registry entries that we didn't archive
        pattern = self._pattern
        for name, mod in list(modules.items()):
            if pattern.match(name) and name not in backup:
                reg = getattr(mod, "__warningregistry__", None)
                if reg:
                    reg.clear()


if __name__ == '__main__':
    unittest.main()
