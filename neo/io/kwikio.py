# -*- coding: utf-8 -*-
"""
Class for reading data from a .kwik dataset

Depends on: scipy
            h5py >= 2.5.0

Supported: Read

Author: Mikkel E. Lepperød @CINPLA

"""
# TODO: units
# TODO: enable reading of all otputfiles
# TODO: enable writing to file
# TODO: stimulus and tracking data
# TODO: enable read all datasets

# needed for python 3 compatibility
from __future__ import absolute_import
from __future__ import division

import numpy as np
import quantities as pq
import os
# version checking
from distutils import version
# check h5py
# try:
#     import h5py
# except ImportError as err:
#     HAVE_H5PY = False
#     H5PY_ERR = err
# else:
#     if version.LooseVersion(h5py.__version__) < '2.5.0':
#         HAVE_H5PY = False
#         H5PY_ERR = ImportError("your h5py version is too old to " +
#                                  "support KwikIO, you need at least 2.5.0 " +
#                                  "You have %s" % h5py.__version__)
#     else:
#         HAVE_H5PY = True
#         H5PY_ERR = None
# check scipy
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

    supported_objects = [Block, Segment, SpikeTrain,
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
        BaseIO.__init__(self)
        self.filename = os.path.abspath(filename)
        model = kwik.KwikModel(self.filename) # TODO this group is loaded twice
        self.models = [kwik.KwikModel(self.filename, channel_group=grp)
                       for grp in model.channel_groups]

    def read_block(self,
                   lazy=False,
                   cascade=True,
                   get_waveforms=True,
                   cluster_metadata='all',
                   ):
        """

        """

        blk = Block()
        if cascade:
            seg = Segment(file_origin=self.filename)
            blk.segments += [seg]
            for model in self.models:
                chx = ChannelIndex(name='channel_group #%i' % model.channel_group,
                                   index=model.channels,
                                   **{'group': model.channel_group})
                blk.channel_indexes.append(chx)
                clusters = model.spike_clusters
                for cluster_id in model.cluster_ids:
                    meta = model.cluster_metadata[cluster_id]
                    if cluster_metadata != 'all':
                        if meta != cluster_metadata:
                            continue
                    sptr = self.read_spiketrain(cluster_id=cluster_id, model=model,
                                           lazy=lazy, cascade=cascade,
                                           get_waveforms=get_waveforms)
                    sptr.annotations.update({'cluster_metadata': meta,
                                             'channel_group': model.channel_group})
                    sptr.channel_index = chx
                    seg.spiketrains.append(sptr)
            seg.duration = model.duration * pq.s

        blk.create_many_to_one_relationship()
        return blk

    def read_spiketrain(self, cluster_id, model,
                        lazy=False,
                        cascade=True,
                        get_waveforms=True,
                        ):
        try:
            if ((not(cluster_id in model.cluster_ids))):
                raise ValueError
        except ValueError:
                print("Exception: cluster_id (%d) not found !! " % cluster_id)
                return
        clusters = model.spike_clusters
        idx = np.argwhere(clusters == cluster_id)
        if get_waveforms:
            w = model.all_waveforms[idx]
            time, spike, channel_index = w.shape
            w = w.reshape(spike, channel_index, time)
        else:
            w = None
        sptr = SpikeTrain(times=model.spike_times[idx],
                          t_stop=model.duration, waveforms=w, units='s',
                          sampling_rate=model.sample_rate,
                          file_origin=self.filename,
                          **{'cluster_id': cluster_id})
        return sptr
