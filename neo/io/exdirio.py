# -*- coding: utf-8 -*-
"""
This is the implementation of the NEO IO for the exdir format.
Depends on: scipy
            h5py >= 2.5.0
            numpy
            quantities
Supported: Read
Authors: Milad H. Mobarhan @CINPLA,
         Svenn-Arne Dragly @CINPLA,
         Mikkel E. Lepperød @CINPLA
"""

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import sys
from neo.io.baseio import BaseIO
from neo.core import (Segment, SpikeTrain, Unit, Epoch, AnalogSignal,
                      ChannelIndex, Block, IrregularlySampledSignal)
import numpy as np
import quantities as pq
import os
import os.path as op
import copy
import shutil

python_version = sys.version_info.major
if python_version == 2:
    from future.builtins import str

try:
    import exdir
    HAVE_EXDIR = True
    EXDIR_ERR = None
except ImportError as err:
    HAVE_EXDIR = False
    EXDIR_ERR = err


class ExdirIO(BaseIO):
    """
    Class for reading/writting of exdir fromat
    """

    is_readable = True
    is_writable = True

    supported_objects = [Block, Segment, AnalogSignal, ChannelIndex, SpikeTrain]
    readable_objects = [Block, SpikeTrain]
    writeable_objects = []

    has_header = False
    is_streameable = False

    name = 'exdir'
    description = 'This IO reads experimental data from an exdir directory'
    extensions = ['exdir']
    # mode can be 'file' or 'dir' or 'fake' or 'database'
    # the main case is 'file' but some reader are base on a directory or a database
    # thinfo is for GUI stuff also
    mode = 'dir'

    def __init__(self, dirname, mode='a'):
        """
        Arguments:
            directory_path : the directory path
        """
        if not HAVE_EXDIR:
            raise EXDIR_ERR
        BaseIO.__init__(self)
        self._absolute_directory_path = dirname
        self._path, relative_directory_path = os.path.split(dirname)
        self._base_directory, extension = os.path.splitext(relative_directory_path)

        if extension != ".exdir":
            raise ValueError("directory extension must be '.exdir'")

        self._exdir_directory = exdir.File(directory=dirname, mode=mode)

        # TODO check if group exists
        self._processing = self._exdir_directory.require_group("processing")
        self._epochs = self._exdir_directory.require_group("epochs")

    def _sptrs_to_times(self, sptrs):
        out = np.array([t for sptr in sptrs
                        for t in sptr.times.rescale('s').magnitude])
        # HACK sometimes out is shape (n_spikes, 1)
        return np.reshape(out, len(out)) * pq.s

    def _sptrs_to_wfseriesf(self, sptrs):
        wfs = np.vstack([sptr.waveforms for sptr in sptrs])
        assert wfs.shape[1:] == sptrs[0].waveforms.shape[1:]
        # neo: num_spikes, num_chans, samples_per_spike = wfs.shape
        return wfs

    def _sptrs_to_spike_clusters(self, sptrs):
        if 'cluster_id' in sptrs[0].annotations:  # assumes its true for all
            out = np.array([i for sptr in sptrs for i in
                           [sptr.annotations['cluster_id']] * len(sptr)])
        else:
            out = np.array([i for j, sptr in enumerate(sptrs)
                            for i in [j] * len(sptr)])
        # HACK sometimes out is shape (n_spikes, 1)
        return np.reshape(out, len(out))

    def write_event_waveform(self, sptrs, path, **annotations):
        channel_group = self._exdir_directory[path]
        sampling_rate = sptrs[0].sampling_rate.rescale('Hz')
        spike_times = self._sptrs_to_times(sptrs)
        waveforms = self._sptrs_to_wfseriesf(sptrs)
        group = channel_group.require_group('EventWaveform')
        wf_group = group.require_group('waveform_timeseries')
        attr = {'num_samples': len(spike_times),
                'sample_length': waveforms.shape[1],
                'sample_rate': sampling_rate}
        attr.update(annotations)
        wf_group.attrs = attr
        ts_data = wf_group.require_dataset("timestamps", data=spike_times)
        ts_data.attrs['num_samples'] = len(spike_times)
        wf = wf_group.require_dataset("data", data=waveforms)
        wf.attrs['num_samples'] = len(spike_times)
        wf.attrs['sample_rate'] = sampling_rate

    def write_clusters(self, sptrs, path, **annotations):
        channel_group = self._exdir_directory[path]
        spike_clusters = self._sptrs_to_spike_clusters(sptrs)
        spike_times = self._sptrs_to_times(sptrs)
        cl_group = channel_group.require_group('Clustering')
        if annotations:
            cl_group.attrs = annotations
        cluster_nums = np.unique(spike_clusters)
        dset = cl_group.require_dataset('nums', data=spike_clusters)
        dset.attrs['num_samples'] = len(spike_times)
        dset = cl_group.require_dataset('cluster_nums', data=cluster_nums)
        dset.attrs['num_samples'] = len(cluster_nums)
        dset = cl_group.require_dataset('timestamps', data=spike_times)
        dset.attrs['num_samples'] = len(spike_times)

    def write_unit_times(self, units, path, **annotations):
        channel_group = self._exdir_directory[path]
        if 'UnitTimes' in channel_group:
            shutil.rmtree(str(channel_group['UnitTimes'].directory))
        unit_times_group = channel_group.require_group('UnitTimes')
        if annotations:
            unit_times_group.attrs = annotations
        for unit_id, unit in enumerate(units):
            if len(unit.spiketrains) > 1:
                raise NotImplementedError('sorry we do not support units ' +
                                          'with several spiketrains')
            sptr = unit.spiketrains[0]
            if 'cluster_id' in sptr.annotations:
                sptr_id = sptr.annotations['cluster_id']
            else:
                sptr_id = unit_id
            unit_group = unit_times_group.require_group('{}'.format(sptr_id))
            unit_attrs = copy.deepcopy(unit.annotations)
            unit.name = unit.name or str(unit_id)
            unit_attrs.update({'name': unit.name,
                               'description': unit.description})
            unit_group.attrs = unit_attrs
            self.write_spiketimes(sptr, unit_group.name)

    def write_spiketimes(self, sptr, path, **annotations):
        channel_group = self._exdir_directory[path]
        sptr_attrs = copy.deepcopy(sptr.annotations)
        sptr_attrs.update(annotations)
        ts_data = channel_group.require_dataset('times', data=sptr.times)
        ts_data.attrs['num_samples'] = len(sptr.times)
        sptr_attrs.update({'name': sptr.name,
                           'description': sptr.description,
                           'start_time': sptr.t_start,
                           'stop_time': sptr.t_stop})
        atr = ts_data.attrs.to_dict()
        sptr_attrs.update(atr)
        ts_data.attrs = sptr_attrs

    def write_analogsignal(self, ana, path, name=None, **annotations):
        channel_group = self._exdir_directory[path]
        description = ana.description or 'AnalogSignal'
        group = channel_group.require_group(description)
        name = name or 'timeseries'
        lfp_group = group.require_group(name)
        attrs = copy.deepcopy(ana.annotations)
        attrs.update(annotations)
        attrs.update({'name': ana.name,
                      'description': ana.description,
                      'start_time': ana.t_start,
                      'stop_time': ana.t_stop,
                      'sample_rate': ana.sampling_rate})
        lfp_group.attrs = attrs
        lfp_data = lfp_group.require_dataset('data', data=ana)
        lfp_data.attrs['num_samples'] = len(ana)
        lfp_data.attrs['sample_rate'] = ana.sampling_rate # TODO not save twice

    def write_epoch(self, epo, path, name=None, **annotations):
        group = self._exdir_directory[path]
        name = name or 'epoch'
        epo_group = group.require_group(name)
        attrs = copy.deepcopy(epo.annotations)
        attrs.update(annotations)
        attrs.update({'name': epo.name, 'description': epo.description})
        dset = epo_group.require_dataset('timestamps', data=epo.times)
        dset.attrs['num_samples'] = len(epo.times)
        if epo.durations is not None:
            dset = epo_group.require_dataset('durations', data=epo.durations)
            dset.attrs['num_samples'] = len(epo.durations)
        dset = epo_group.require_dataset('data', data=epo.labels)
        dset.attrs['num_samples'] = len(epo.labels)
        epo_group.attrs = attrs

    def write_block(self, blk, elphys_directory_name='electrophysiology'):
        if not isinstance(blk, Block):
            raise ValueError('Input must be of type neo.Block')
        if len(blk.segments) > 1:
            raise ValueError('sorry, exdir supports only one segment')
        seg = blk.segments[0]
        annotations = copy.deepcopy(blk.annotations)
        annotations.update({'session_duration': seg.t_stop - seg.t_start})
        self._exdir_directory.attrs = annotations
        for epo_num, epo in enumerate(seg.epochs):
            name = 'Epoch_{}'.format(epo_num)
            self.write_epoch(epo, self._epochs.name, name=name,
                             start_time=seg.t_start, stop_time=seg.t_stop)
        elphys = self._processing.require_group(elphys_directory_name)
        for chx in blk.channel_indexes:
            self.write_channelindex(chx, elphys.name, start_time=seg.t_start,
                                    stop_time=seg.t_stop)
        if len(blk.channel_indexes) == 0:
            self.write_channelindex(chx, elphys.name, start_time=seg.t_start,
                                    stop_time=seg.t_stop)

    def write_segment(self, seg, path, **annotations):
        group = self._exdir_directory[path]
        segment_group = group.require_group('segment_0')
        attrs = {'segment_id': seg.index}
        attrs.update(annotations)
        segment_group.attrs = attrs
        for idx, ana in enumerate(seg.analogsignals):
            self.write_analogsignal(ana, segment_group.name,
                                    name='timeseries_{}'.format(idx),
                                    **attrs)
        sptrs = [st for st in seg.spiketrains]
        if len(sptrs) == 0:
            return
        if sptrs[0].waveforms is not None:
            self.write_event_waveform(sptrs, segment_group.name, **attrs)
            self.write_clusters(sptrs, segment_group.name, **attrs)
        units = [unit for unit in chx.units]
        self.write_unit_times(units, segment_group.name, **attrs)

    def write_channelindex(self, chx, path=None,
                           elphys_directory_name='electrophysiology',
                           **annotations):
        path = path or self._processing.require_group(elphys_directory_name).name
        group = self._exdir_directory[path]
        if 'group_id' in chx.annotations:
            group_id = chx.annotations['group_id']
            group_name = 'channel_group_{}'.format(group_id)
        else:
            print('Warning: "group_id" is not in channel_index.annotations ' +
                  'setting electrode_group_id as None and group name ' +
                  'as unique string')
            group_id = None
            cnt = 0
            while True:
                group_name = 'channel_group_{}'.format(cnt)
                if group_name in group.keys():
                    cnt += 1
                else:
                    break

        channel_group = group.require_group(group_name)
        attrs = {'electrode_idx': chx.index,
                 'electrode_group_id': group_id,
                 'electrode_identities': chx.channel_ids}
        attrs.update(annotations)
        channel_group.attrs = attrs
        for idx, ana in enumerate(chx.analogsignals):
            self.write_analogsignal(ana, channel_group.name,
                                    name='timeseries_{}'.format(idx),
                                    **attrs)
        sptrs = [st for unit in chx.units for st in unit.spiketrains]
        if len(sptrs) == 0:
            return
        if sptrs[0].waveforms is not None:
            self.write_event_waveform(sptrs, channel_group.name, **attrs)
            self.write_clusters(sptrs, channel_group.name, **attrs)
        units = [unit for unit in chx.units]
        self.write_unit_times(units, channel_group.name, **attrs)

    def read_block(self,
                   lazy=False,
                   cascade=True,
                   read_waveforms=True,
                   elphys_directory_name='electrophysiology'):
        '''

        '''
        blk = Block(file_origin=self._absolute_directory_path,
                    **self._exdir_directory.attrs.to_dict())
        seg = Segment(name='Segment #0', index=0)
        blk.segments.append(seg)
        if cascade:
            for group in self._epochs.values():
                epo = self.read_epoch(group.name, cascade, lazy)
                seg.epochs.append(epo)
            for channel_group in self._processing[elphys_directory_name].values():
                chx = self.read_channelindex(channel_group.name,
                                             cascade=cascade,
                                             lazy=lazy,
                                             read_waveforms=read_waveforms)
                blk.channel_indexes.append(chx)
                seg.analogsignals.extend(chx.analogsignals)
                seg.spiketrains.extend([sptr for unit in chx.units
                                        for sptr in unit.spiketrains])
        return blk

    def read_segment(self, cascade=True, lazy=False, read_waveforms=True):
        seg = Segment(name='Segment #0', index=0)
        if cascade:
            for group in self._epochs.values():
                epo = self.read_epoch(group.name, cascade, lazy)
                seg.epochs.append(epo)

            for prc_group in self._processing.values():
                for sub_group in prc_group.values():
                    if 'LFP' in sub_group:
                        for lfp_group in sub_group['LFP'].values():
                            ana = self.read_analogsignal(lfp_group.name,
                                                         cascade=cascade,
                                                         lazy=lazy)
                            seg.analogsignals.append(ana)

                    sptr = None
                    if 'UnitTimes' in sub_group:
                        for unit_group in sub_group['UnitTimes'].values():
                            sptr = self.read_spiketrain(
                                unit_group.name,
                                cascade=cascade,
                                lazy=lazy,
                                read_waveforms=read_waveforms
                            )
                            seg.spiketrains.append(sptr)
                    elif 'EventWaveform' in sub_group:
                        sptr = self.read_spiketrain(
                            sub_group['UnitTimes'].name,
                            cascade=cascade,
                            lazy=lazy,
                            read_waveforms=read_waveforms
                        )
                    if sptr is not None:
                        seg.spiketrains.append(sptr)
        return seg

    def read_channelindex(self, path, cascade=True, lazy=False,
                          read_waveforms=True):
        channel_group = self._exdir_directory[path]
        group_id = channel_group.attrs['electrode_group_id']
        chx = ChannelIndex(
            name='Channel group {}'.format(group_id),
            index=channel_group.attrs['electrode_idx'],
            channel_ids=channel_group.attrs['electrode_identities'],
            **{'group_id': group_id,
               'exdir_path': path}
        )
        if 'LFP' in channel_group:
            for lfp_group in channel_group['LFP'].values():
                ana = self.read_analogsignal(lfp_group.name,
                                             cascade=cascade,
                                             lazy=lazy)
                chx.analogsignals.append(ana)
                ana.channel_index = chx
        sptrs = []
        if 'UnitTimes' in channel_group:
            for unit_group in channel_group['UnitTimes'].values():
                unit = self.read_unit(
                    unit_group.name,
                    cascade=cascade,
                    lazy=lazy,
                    read_waveforms=read_waveforms
                )
                unit.channel_index = chx
                chx.units.append(unit)
                sptr = unit.spiketrains[0]
                sptr.channel_index = chx

        elif 'EventWaveform' in channel_group:
            sptr = self.read_spiketrain(
                channel_group['EventWaveform'].name,
                cascade=cascade,
                lazy=lazy,
                read_waveforms=read_waveforms
            )
            unit = Unit(name=sptr.name,
                        **sptr.annotations)
            unit.spiketrains.append(sptr)
            unit.channel_index = chx
            sptr.channel_index = chx
            chx.units.append(unit)
        return chx

    def read_epoch(self, path, cascade=True, lazy=False):
        group = self._exdir_directory[path]
        if lazy:
            times = []
        else:
            times = pq.Quantity(group['timestamps'].data,
                                group['timestamps'].attrs['unit'])

        if "durations" in group and not lazy:
            durations = pq.Quantity(group['durations'].data, group['durations'].attrs['unit'])
        elif "durations" in group and lazy:
            durations = []
        else:
            durations = None

        if 'data' in group and not lazy:
            if 'unit' not in group['data'].attrs:
                labels = group['data'].data
            else:
                labels = pq.Quantity(group['data'].data,
                                     group['data'].attrs['unit'])
        elif 'data' in group and lazy:
            labels = []
        else:
            labels = None
        annotations = {'exdir_path': path}
        annotations.update(group.attrs.to_dict())
        if lazy:
            lazy_shape = (group.attrs['num_samples'],)
        else:
            lazy_shape = None
        epo = Epoch(times=times, durations=durations, labels=labels,
                    lazy_shape=lazy_shape, **annotations)

        return epo

    def read_analogsignal(self, path, cascade=True, lazy=False):
        group = self._exdir_directory[path]
        signal = group["data"]
        attrs = {'exdir_path': path}
        attrs.update(group.attrs.to_dict())
        if lazy:
            ana = AnalogSignal([],
                               lazy_shape=(signal.attrs["num_samples"],),
                               units=signal.attrs["unit"],
                               sampling_rate=group.attrs['sample_rate'],
                               **attrs)
        else:
            ana = AnalogSignal(signal.data,
                               units=signal.attrs["unit"],
                               sampling_rate=group.attrs['sample_rate'],
                               **attrs)
        return ana

    def read_unit(self, path, cascade=True, lazy=False, cluster_num=None,
                  read_waveforms=True):
        group = self._exdir_directory[path]
        assert group.parent.object_name == 'UnitTimes'
        attrs = {'exdir_path': path}
        attrs.update(group.attrs.to_dict())
        unit = Unit(**attrs)
        sptr = self.read_spiketrain(path, cascade, lazy, cluster_num,
                                    read_waveforms)
        unit.spiketrains.append(sptr)
        return unit

    def read_spiketrain(self, path, cascade=True, lazy=False, cluster_num=None,
                        read_waveforms=True):
        group = self._exdir_directory[path]
        metadata = {}
        if group.parent.object_name == 'UnitTimes':
            if lazy:
                times = [] * pq.s
            else:
                times = pq.Quantity(group['times'].data,
                                    group['times'].attrs['unit'])
            t_stop = group.parent.attrs['stop_time']
            t_start = group.parent.attrs['start_time']
            metadata.update(group['times'].attrs.to_dict())
            if read_waveforms:
                if not 'EventWaveform' in group.parent.parent:
                    raise ValueError('No EventWaveform detected in exdir ' +
                                     'directory, please revise directory ' +
                                     'or set read_waveforms to False')
                wf_group = group.parent.parent['EventWaveform']
                cluster_group = group.parent.parent['Clustering']
                cluster_num = cluster_num or int(group.object_name.split("_")[-1])
                cluster_ids = cluster_group['nums'].data
                indices, = np.where(cluster_ids == cluster_num)
        elif group.object_name == 'EventWaveform':
            sub_group = list(group.values())[0]
            # TODO assert all timestamps to be equal if several waveform_timeseries exists
            wf_group = group
            if 'Clustering' in group.parent:
                cluster_group = group.parent['Clustering']
                cluster_ids = cluster_group['num'].data
                if len(np.unique(cluster_ids)) > 1:
                    assert cluster_num is not None, 'You must set cluster_num'
                else:
                    cluster_num = cluster_num or int(np.unique(cluster_ids))
                indices, = np.where(cluster_ids == cluster_num)
                if lazy:
                    times = [] * pq.s
                else:
                    times = pq.Quantity(sub_group["timestamps"].data[indices],
                                        sub_group["timestamps"].attrs['unit'])
            else:
                if lazy:
                    times = [] * pq.s
                else:
                    times = pq.Quantity(sub_group["timestamps"].data,
                                        sub_group["timestamps"].attrs['unit'])
                    indices = range(len(times))
            t_stop = sub_group.attrs['stop_time']
            t_start = sub_group.attrs['start_time']
            metadata.update(sub_group.attrs.to_dict())
        else:
            raise ValueError('Expected a sub group of UnitTimes or an ' +
                             'EventWaveform group')
        if read_waveforms:
            waveforms = []
            for wf in wf_group.values():
                if lazy:
                    data = [] * pq.dimensionless
                else:
                    data = pq.Quantity(wf["data"].data[indices, :, :],
                                       wf["data"].attrs['unit'])
                waveforms.append(data)
                metadata.update(wf.attrs.to_dict())
            waveforms = np.vstack(waveforms)
            # TODO assert shape of waveforms relative to channel_ids etc
            sampling_rate = wf["data"].attrs['sample_rate']
        else:
            waveforms = None
            sampling_rate = None
        metadata.update({'exdir_path': path})
        sptr = SpikeTrain(times=times,
                          t_stop=t_stop,
                          t_start=t_start,
                          waveforms=waveforms,
                          sampling_rate=sampling_rate,
                          **metadata)
        return sptr
