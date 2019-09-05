# -*- coding: utf-8 -*-
"""
Class for reading data from a .kwik dataset

Depends on: scipy
            phy

Supported: Read

Author: Mikkel E. Lepperød @CINPLA

"""
# TODO: writing to file

# needed for python 3 compatibility
from __future__ import absolute_import
from __future__ import division

import numpy as np
import quantities as pq
import os

try:
    from scipy import stats
except ImportError as err:
    HAVE_SCIPY = False
    SCIPY_ERR = err
else:
    HAVE_SCIPY = True
    SCIPY_ERR = None

try:
    from klusta import kwik
except ImportError as err:
    HAVE_KWIK = False
    KWIK_ERR = err
else:
    HAVE_KWIK = True
    KWIK_ERR = None

# I need to subclass BaseIO
from neo.io.baseio import BaseIO

# to import from core
from neo.core import (Segment, SpikeTrain, Unit, Epoch, AnalogSignal,
                      ChannelIndex, Block)
import neo.io.tools


class KwikIO(BaseIO):
    """
    Class for "reading" experimental data from a .kwik file.

    Generates a :class:`Segment` with a :class:`AnalogSignal`

    """

    is_readable = True  # This class can only read data
    is_writable = False  # write is not supported

    supported_objects = [Block, Segment, SpikeTrain, AnalogSignal,
                         ChannelIndex]

    # This class can return either a Block or a Segment
    # The first one is the default ( self.read )
    # These lists should go from highest object to lowest object because
    # common_io_test assumes it.
    readable_objects = [Block]

    # This class is not able to write objects
    writeable_objects = []

    has_header = False
    is_streameable = False

    name = 'Kwik'
    description = 'This IO reads experimental data from a .kwik dataset'
    extensions = ['kwik']
    mode = 'file'

    def __init__(self, filename):
        """
        Arguments:
            filename : the filename
        """
        if not HAVE_KWIK:
            raise KWIK_ERR
        BaseIO.__init__(self)
        self.filename = os.path.abspath(filename)
        model = kwik.KwikModel(self.filename)  # TODO this group is loaded twice
        self.models = [kwik.KwikModel(self.filename, channel_group=grp)
                       for grp in model.channel_groups]

    def read_block(self,
                   lazy=False,
                   get_waveforms=True,
                   cluster_group=None,
                   raw_data_units='uV',
                   get_raw_data=False,
                   ):
        """
        Reads a block with segments and channel_indexes

        Parameters:
        get_waveforms: bool, default = False
            Wether or not to get the waveforms
        get_raw_data: bool, default = False
            Wether or not to get the raw traces
        raw_data_units: str, default = "uV"
            SI units of the raw trace according to voltage_gain given to klusta
        cluster_group: str, default = None
            Which clusters to load, possibilities are "noise", "unsorted",
            "good", if None all is loaded.
        """
        assert not lazy, 'Do not support lazy'

        blk = Block()
        seg = Segment(file_origin=self.filename)
        blk.segments += [seg]
        for model in self.models:
            group_id = model.channel_group
            group_meta = {'group_id': group_id}
            group_meta.update(model.metadata)
            chx = ChannelIndex(name='channel group #{}'.format(group_id),
                               index=model.channels,
                               **group_meta)
            blk.channel_indexes.append(chx)
            clusters = model.spike_clusters
            for cluster_id in model.cluster_ids:
                meta = model.cluster_metadata[cluster_id]
                if cluster_group is None:
                    pass
                elif cluster_group != meta:
                    continue
                sptr = self.read_spiketrain(cluster_id=cluster_id,
                                            model=model,
                                            get_waveforms=get_waveforms,
                                            raw_data_units=raw_data_units)
                sptr.annotations.update({'cluster_group': meta,
                                         'group_id': model.channel_group})
                sptr.channel_index = chx
                unit = Unit(cluster_group=meta,
                            group_id=model.channel_group,
                            name='unit #{}'.format(cluster_id))
                unit.spiketrains.append(sptr)
                chx.units.append(unit)
                unit.channel_index = chx
                seg.spiketrains.append(sptr)
            if get_raw_data:
                ana = self.read_analogsignal(model, units=raw_data_units)
                ana.channel_index = chx
                seg.analogsignals.append(ana)

        seg.duration = model.duration * pq.s

        blk.create_many_to_one_relationship()
        return blk

    def read_analogsignal(self, model, units='uV', lazy=False):
        """
        Reads analogsignals

        Parameters:
        units: str, default = "uV"
            SI units of the raw trace according to voltage_gain given to klusta
        """
        assert not lazy, 'Do not support lazy'

        arr = model.traces[:] * model.metadata['voltage_gain']
        ana = AnalogSignal(arr, sampling_rate=model.sample_rate * pq.Hz,
                           units=units,
                           file_origin=model.metadata['raw_data_files'])
        return ana

    def read_spiketrain(self, cluster_id, model,
                        lazy=False,
                        get_waveforms=True,
                        raw_data_units=None
                        ):
        """
        Reads sorted spiketrains

        Parameters:
        get_waveforms: bool, default = False
            Wether or not to get the waveforms
        cluster_id: int,
            Which cluster to load, according to cluster id from klusta
        model: klusta.kwik.KwikModel
            A KwikModel object obtained by klusta.kwik.KwikModel(fname)
        """
        try:
            if ((not (cluster_id in model.cluster_ids))):
                raise ValueError
        except ValueError:
            print("Exception: cluster_id (%d) not found !! " % cluster_id)
            return
        clusters = model.spike_clusters
        idx = np.nonzero(clusters == cluster_id)
        if get_waveforms:
            w = model.all_waveforms[idx]
            # klusta: num_spikes, samples_per_spike, num_chans = w.shape
            w = w.swapaxes(1, 2)
            w = pq.Quantity(w, raw_data_units)
        else:
            w = None
        sptr = SpikeTrain(times=model.spike_times[idx],
                          t_stop=model.duration, waveforms=w, units='s',
                          sampling_rate=model.sample_rate * pq.Hz,
                          file_origin=self.filename,
                          **{'cluster_id': cluster_id})
        return sptr
