# -*- coding: utf-8 -*-
"""
Tests of io.exdirio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.core import (Block, Segment, ChannelIndex, AnalogSignal,
                      Unit, SpikeTrain, Epoch)
from neo.test.iotest.common_io_test import BaseTestIO

try:
    import exdir
    HAVE_EXDIR = True
except ImportError:
    HAVE_EXDIR = False

from neo.io.exdirio import ExdirIO
import shutil
import numpy as np
import quantities as pq
import os


@unittest.skipUnless(HAVE_EXDIR, "Requires exdir")
class TestExdirIO(unittest.TestCase):

    def setUp(self):
        self.fname = '/tmp/test.exdir'
        if os.path.exists(self.fname):
            shutil.rmtree(self.fname)
        self.n_channels = 5
        self.n_samples = 20
        self.n_spikes = 50
        blk = Block()
        seg = Segment()
        blk.segments.append(seg)
        chx1 = ChannelIndex(index=np.arange(self.n_channels),
                            channel_ids=np.arange(self.n_channels))
        chx2 = ChannelIndex(index=np.arange(self.n_channels),
                            channel_ids=np.arange(self.n_channels) * 2)
        blk.channel_indexes.extend([chx1, chx2])

        wf1 = np.random.random((self.n_spikes, self.n_channels,
                                self.n_samples))
        ts1 = np.sort(np.random.random(self.n_spikes))
        t_stop1 = np.ceil(ts1[-1])
        sptr1 = SpikeTrain(times=ts1, units='s',
                           waveforms=np.random.random((self.n_spikes,
                                                       self.n_channels,
                                                       self.n_samples)) * pq.V,
                           name='spikes 1',
                           description='sptr1',
                           t_stop=t_stop1, **{'id': 1})
        sptr1.channel_index = chx1
        unit1 = Unit(name='unit 1')
        unit1.spiketrains.append(sptr1)
        chx1.units.append(unit1)
        seg.spiketrains.append(sptr1)

        ts2 = np.sort(np.random.random(self.n_spikes))
        t_stop2 = np.ceil(ts2[-1])
        sptr2 = SpikeTrain(times=ts2, units='s',
                           waveforms=np.random.random((self.n_spikes,
                                                       self.n_channels,
                                                       self.n_samples)) * pq.V,
                           description='sptr2',
                           name='spikes 2',
                           t_stop=t_stop2, **{'id': 2})
        sptr2.channel_index = chx2
        unit2 = Unit(name='unit 2')
        unit2.spiketrains.append(sptr2)
        chx2.units.append(unit2)
        seg.spiketrains.append(sptr2)

        wf3 = np.random.random((self.n_spikes, self.n_channels, self.n_samples))
        ts3 = np.sort(np.random.random(self.n_spikes))
        t_stop3 = np.ceil(ts3[-1])
        sptr3 = SpikeTrain(times=ts3, units='s',
                           waveforms=np.random.random((self.n_spikes,
                                                       self.n_channels,
                                                       self.n_samples)) * pq.V,
                           description='sptr3',
                           name='spikes 3',
                           t_stop=t_stop3, **{'id': 3})
        sptr3.channel_index = chx2
        unit3 = Unit(name='unit 3')
        unit3.spiketrains.append(sptr3)
        chx2.units.append(unit3)
        seg.spiketrains.append(sptr3)

        t_stop = max([t_stop1, t_stop2, t_stop3]) * pq.s

        ana = AnalogSignal(np.random.random(self.n_samples),
                           sampling_rate=self.n_samples / t_stop,
                           units='V',
                           name='ana1',
                           description='LFP')
        assert t_stop == ana.t_stop
        seg.analogsignals.append(ana)
        epo = Epoch(np.random.random(self.n_samples),
                    durations=[1] * self.n_samples * pq.s,
                    units='s',
                    name='epo1')
        seg.epochs.append(epo)
        self.blk = blk

    def tearDown(self):
        if os.path.exists(self.fname):
            shutil.rmtree(self.fname)
    
    def test_write_clusters(self):
        io = ExdirIO(self.fname)
        ex = io._processing.require_group('electrophysiology')
        ex = ex.require_group('elgroup')
        io.write_clusters([sptr for sptr in self.blk.segments[0].spiketrains],
                          ex.name, test_key=4)
        assert ex['Clustering'].attrs['test_key'] == 4
    
    def test_write_event_waveform(self):
        io = ExdirIO(self.fname)
        ex = io._processing.require_group('electrophysiology')
        ex = ex.require_group('elgroup')
        sptrs = [sptr for sptr in self.blk.segments[0].spiketrains]
        io.write_event_waveform(sptrs, ex.name, test_key=4)
        wfgroup = ex['EventWaveform']['waveform_timeseries']
        assert wfgroup.attrs['test_key'] == 4
        assert wfgroup['data'].data.shape == (3 * self.n_spikes,
                                              self.n_channels,
                                              self.n_samples
                                              ), wfgroup['data'].data.shape
    
    def test_write_spiketimes(self):
        io = ExdirIO(self.fname)
        ex = io._processing.require_group('electrophysiology')
        ex = ex.require_group('elgroup').require_group('UnitTimes')
        ex = ex.require_group('0')
        sptr = self.blk.segments[0].spiketrains[0]
        io.write_spiketimes(sptr, ex.name, test_key=4)
        assert ex['times'].attrs['test_key'] == 4
        assert ex['times'].attrs['name'] == sptr.name
        assert ex['times'].attrs['description'] == sptr.description
        assert ex['times'].attrs['start_time'] == sptr.t_start
        assert ex['times'].attrs['stop_time'] == sptr.t_stop
        
    def test_write_unit_times(self):
        io = ExdirIO(self.fname)
        ex = io._processing.require_group('electrophysiology')
        ex = ex.require_group('elgroup')
        units = [unit for chx in self.blk.channel_indexes for unit in chx.units]
        unit_dict = {unit.name: unit.spiketrains[0] for unit in units}
        io.write_unit_times(units, ex.name, test_key=4)
        unit_times_group = ex['UnitTimes']
        assert unit_times_group.attrs['test_key'] == 4
        for unit_group in unit_times_group.values():
            name = unit_group.attrs['name']
            np.testing.assert_array_equal(unit_dict[name].times,
                                          unit_group['times'].data)
            sptr = unit_dict[name]
            assert unit_group['times'].attrs['name'] == sptr.name
            assert unit_group['times'].attrs['description'] == sptr.description
            assert unit_group['times'].attrs['start_time'] == sptr.t_start
            assert unit_group['times'].attrs['stop_time'] == sptr.t_stop
                                              
    def test_write_epoch(self):
        io = ExdirIO(self.fname)
        epo = self.blk.segments[0].epochs[0]
        io.write_epoch(epo, io._epochs.name, 'epo1')
        io.write_epoch(epo, io._epochs.name, 'epo2')
        np.testing.assert_array_equal(io._epochs['epo1']['timestamps'].data,
                                      epo.times)
        np.testing.assert_array_equal(io._epochs['epo1']['durations'].data,
                                      epo.durations)
        np.testing.assert_array_equal(io._epochs['epo1']['data'].data,
                                      epo.labels)
    
    def test_write_analogsignal(self):
        io = ExdirIO(self.fname)
        ex = io._processing.require_group('electrophysiology')
        ex = ex.require_group('elgroup')
        ana = self.blk.segments[0].analogsignals[0]
        io.write_analogsignal(ana, ex.name, 'timeseries')
        ana_group = ex[ana.description]['timeseries']
        np.testing.assert_array_equal(ana_group['data'].data, ana.magnitude)
        np.testing.assert_equal(ana_group.attrs['start_time'], ana.t_start)
        np.testing.assert_equal(ana_group.attrs['stop_time'], ana.t_stop)
        np.testing.assert_equal(ana_group.attrs['sample_rate'],
                                ana.sampling_rate)
    
    def test_write_channelindex(self):
        io = ExdirIO(self.fname)
        elphys = io._processing.require_group('electrophysiology')
        chxs = {tuple(chx.channel_ids): chx for chx in self.blk.channel_indexes}
        [io.write_channelindex(chx, elphys.name) for chx in chxs.values()]
        save_chxs = {tuple(chgrp.attrs['electrode_identities']): chgrp
                     for chgrp in elphys.values()}
        np.testing.assert_equal(len(chxs), len(save_chxs))
        for key, chx in chxs.items():
            ex = save_chxs[key]
            for ana in chx.analogsignals:
                ana_group = ex[ana.description]['timeseries']
                np.testing.assert_array_equal(ana_group['data'].data, ana.magnitude)
                np.testing.assert_equal(ana_group.attrs['start_time'], ana.t_start)
                np.testing.assert_equal(ana_group.attrs['stop_time'], ana.t_stop)
                np.testing.assert_equal(ana_group.attrs['sample_rate'],
                                        ana.sampling_rate)
            
            unit_times_group = ex['UnitTimes']
            unit_dict = {unit.name: unit.spiketrains[0] for unit in chx.units}
            for unit in chx.units:
                for unit_group in unit_times_group.values():
                    name = unit_group.attrs['name']
                    np.testing.assert_array_equal(unit_dict[name].times,
                                                  unit_group['times'].data)
                    sptr = unit_dict[name]
                    assert unit_group['times'].attrs['name'] == sptr.name
                    assert unit_group['times'].attrs['description'] == sptr.description
                    assert unit_group['times'].attrs['start_time'] == sptr.t_start
                    assert unit_group['times'].attrs['stop_time'] == sptr.t_stop
            sptrs = [sptr for sptr in unit_dict.values()]
            wfgroup = ex['EventWaveform']['waveform_timeseries']
            assert wfgroup['data'].data.shape == (len(sptrs) * self.n_spikes,
                                                  self.n_channels,
                                                  self.n_samples
                                                  ), wfgroup['data'].data.shape
    
    def test_write_block(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk, elphys_directory_name='elphys')
        elphys = io._processing['elphys']
        chxs = {tuple(chx.channel_ids): chx for chx in self.blk.channel_indexes}
        save_chxs = {tuple(chgrp.attrs['electrode_identities']): chgrp
                     for chgrp in elphys.values()}
        np.testing.assert_equal(len(chxs), len(save_chxs))
        for key, chx in chxs.items():
            ex = save_chxs[key]
            for ana in chx.analogsignals:
                ana_group = ex[ana.description]['timeseries']
                np.testing.assert_array_equal(ana_group['data'].data, ana.magnitude)
                np.testing.assert_equal(ana_group.attrs['start_time'], ana.t_start)
                np.testing.assert_equal(ana_group.attrs['stop_time'], ana.t_stop)
                np.testing.assert_equal(ana_group.attrs['sample_rate'],
                                        ana.sampling_rate)
            
            unit_times_group = ex['UnitTimes']
            unit_dict = {unit.name: unit.spiketrains[0] for unit in chx.units}
            for unit in chx.units:
                for unit_group in unit_times_group.values():
                    name = unit_group.attrs['name']
                    np.testing.assert_array_equal(unit_dict[name].times,
                                                  unit_group['times'].data)
                    sptr = unit_dict[name]
                    assert unit_group['times'].attrs['name'] == sptr.name
                    assert unit_group['times'].attrs['description'] == sptr.description
                    assert unit_group['times'].attrs['start_time'] == sptr.t_start
                    assert unit_group['times'].attrs['stop_time'] == sptr.t_stop
            sptrs = [sptr for sptr in unit_dict.values()]
            wfgroup = ex['EventWaveform']['waveform_timeseries']
            assert wfgroup['data'].data.shape == (len(sptrs) * self.n_spikes,
                                                  self.n_channels,
                                                  self.n_samples
                                                  ), wfgroup['data'].data.shape
    
    def test_write_read_block(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        io = ExdirIO(self.fname)
        blk = io.read_block()
        np.testing.assert_equal(len(blk.segments[0].spiketrains),
                                len(self.blk.segments[0].spiketrains))
    
    def test_write_read_block_sptr_equal(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        io = ExdirIO(self.fname)
        blk = io.read_block()
        sptrs = {sptr.annotations['id']: sptr
                 for sptr in self.blk.segments[0].spiketrains}
        sptrs_load = {sptr.annotations['id']: sptr
                      for sptr in blk.segments[0].spiketrains}
        for key in sptrs.keys():
            np.testing.assert_array_equal(sptrs[key], sptrs_load[key])
            np.testing.assert_equal(sptrs[key].name, sptrs_load[key].name)
            np.testing.assert_equal(sptrs[key].description,
                                    sptrs_load[key].description)
            for k, v in sptrs[key].annotations.items():
                if k == 'description' or k == 'name':
                    continue
                np.testing.assert_equal(v, sptrs_load[key].annotations[k])
            np.testing.assert_array_equal(sptrs[key].channel_index.index,
                                    sptrs_load[key].channel_index.index)
    
    def test_write_read_block_chxs_equal(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        io = ExdirIO(self.fname)
        blk = io.read_block()
        np.testing.assert_equal(len(self.blk.channel_indexes),
                                len(blk.channel_indexes))
    
    def test_write_read_block_units_equal(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        io = ExdirIO(self.fname)
        blk = io.read_block()
        units = {unit.name: unit
                 for unit in self.blk.channel_indexes[0].units}
        units_load = {unit.name: unit
                      for unit in blk.channel_indexes[0].units}
        for key in units.keys():
            np.testing.assert_equal(len(units[key].spiketrains[0]),
                                    len(units_load[key].spiketrains[0]))
            sptr = units[key].spiketrains[0]
            sptr_load = units_load[key].spiketrains[0]
            np.testing.assert_array_equal(sptr, sptr_load)
            np.testing.assert_equal(units[key].name, units_load[key].name)
    
    def test_write_read_block_epo_equal(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        io = ExdirIO(self.fname)
        blk = io.read_block()
        epos = {epo.name: epo for epo in self.blk.segments[0].epochs}
        epos_load = {epo.name: epo for epo in blk.segments[0].epochs}
        for key in epos.keys():
            np.testing.assert_array_equal(epos[key], epos_load[key])
            np.testing.assert_array_equal(epos[key].durations,
                                          epos_load[key].durations)
            np.testing.assert_equal(epos[key].name, epos_load[key].name)
    
    def test_write_read_block_ana_equal(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        exdir_dir = exdir.File(self.fname)
        lfp_group = exdir_dir['/processing/electrophysiology/channel_group_0']
        ana = self.blk.segments[0].analogsignals[0]
        io.write_analogsignal(ana, lfp_group.name)
        io = ExdirIO(self.fname)
        blk = io.read_block()
        anas = {ana.name: ana for ana in self.blk.segments[0].analogsignals}
        anas_load = {ana.name: ana for ana in blk.segments[0].analogsignals}
        for key in anas.keys():
            np.testing.assert_array_equal(anas[key], anas_load[key])
            np.testing.assert_equal(anas[key].name, anas_load[key].name)


if __name__ == "__main__":
    unittest.main()
