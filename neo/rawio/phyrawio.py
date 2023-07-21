"""
PhyRawIO is a class to handle Phy spike sorting data.
Ported from:
https://github.com/SpikeInterface/spikeextractors/blob/
f20b1219eba9d3330d5d7cd7ce8d8924a255b8c2/spikeextractors/
extractors/phyextractors/phyextractors.py

Author: Regimantas Jurkus
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)

import numpy as np
from pathlib import Path
import re
import csv
import ast
import warnings


class PhyRawIO(BaseRawIO):
    """
    Class for reading Phy data.

    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.PhyRawIO(dirname='/dir/to/phy/folder')
        >>> r.parse_header()
        >>> print(r)
        >>> spike_timestamp = r.get_spike_timestamps(block_index=0,
        ... seg_index=0, spike_channel_index=0, t_start=None, t_stop=None)
        >>> spike_times = r.rescale_spike_timestamp(spike_timestamp, 'float64')

    """
    # file formats used by phy
    extensions = ['npy', 'mat', 'tsv', 'dat']
    rawmode = 'one-dir'

    def __init__(self, dirname='', load_amplitudes=False, load_pcs=False):
        BaseRawIO.__init__(self)
        self.dirname = dirname
        self.load_pcs = load_pcs
        self.load_amplitudes = load_amplitudes

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        phy_folder = Path(self.dirname)

        self._spike_times = np.load(phy_folder / 'spike_times.npy')
        self._spike_templates = np.load(phy_folder / 'spike_templates.npy')

        if (phy_folder / 'spike_clusters.npy').is_file():
            self._spike_clusters = np.load(phy_folder / 'spike_clusters.npy')
        else:
            self._spike_clusters = self._spike_templates

        self._amplitudes = None
        if self.load_amplitudes:
            if (phy_folder / 'amplitudes.npy').is_file():
                self._amplitudes = np.squeeze(np.load(phy_folder / 'amplitudes.npy'))
            else:
                warnings.warn('Amplitudes requested but "amplitudes.npy"'
                              'not found in the data folder.')

        self._pc_features = None
        self._pc_feature_ind = None
        if self.load_pcs:
            if ((phy_folder / 'pc_features.npy').is_file()
                    and (phy_folder / 'pc_feature_ind.npy').is_file()):
                self._pc_features = np.squeeze(np.load(phy_folder / 'pc_features.npy'))
                self._pc_feature_ind = np.squeeze(np.load(phy_folder / 'pc_feature_ind.npy'))
            else:
                warnings.warn('PCs requested but "pc_features.npy" and/or'
                              '"pc_feature_ind.npy" not found in the data folder.')

        # SEE: https://stackoverflow.com/questions/4388626/
        #  python-safe-eval-string-to-bool-int-float-none-string
        if (phy_folder / 'params.py').is_file():
            with (phy_folder / 'params.py').open('r') as f:
                contents = f.read()
            metadata = dict()
            contents = contents.replace('\n', ' ')
            pattern = re.compile(r'(\S*)[\s]?=[\s]?(\S*)')
            elements = pattern.findall(contents)
            for key, value in elements:
                metadata[key.lower()] = ast.literal_eval(value)

        self._sampling_frequency = metadata['sample_rate']

        clust_ids = np.unique(self._spike_clusters)
        self.unit_labels = list(clust_ids)

        self._t_start = 0.
        self._t_stop = max(self._spike_times).item() / self._sampling_frequency

        signal_streams = []
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        signal_channels = []
        signal_channels = np.array(signal_channels,
                                   dtype=_signal_channel_dtype)

        spike_channels = []
        for i, clust_id in enumerate(clust_ids):
            unit_name = f'unit {clust_id}'
            unit_id = f'{clust_id}'
            wf_units = ''
            wf_gain = 0
            wf_offset = 0.
            wf_left_sweep = 0
            wf_sampling_rate = 0
            spike_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                  wf_offset, wf_left_sweep, wf_sampling_rate))
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

        csv_tsv_files = [x for x in phy_folder.iterdir() if
                         x.suffix == '.csv' or x.suffix == '.tsv']

        # annotation_lists is list of list of dict (python==3.8)
        # or list of list of ordered dict (python==3.6)
        # SEE: https://docs.python.org/3/library/csv.html#csv.DictReader
        annotation_lists = [self._parse_tsv_or_csv_to_list_of_dict(file)
                            for file in csv_tsv_files]

        bl_ann = self.raw_annotations['blocks'][0]
        bl_ann['name'] = "Block #0"
        seg_ann = bl_ann['segments'][0]
        seg_ann['name'] = 'Seg #0 Block #0'
        for index, clust_id in enumerate(clust_ids):
            spiketrain_an = seg_ann['spikes'][index]

            # Add cluster_id annotation
            spiketrain_an['cluster_id'] = clust_id

            # Loop over list of list of dict and annotate each st
            for annotation_list in annotation_lists:
                clust_key, *property_names = tuple(annotation_list[0].keys())
                for property_name in property_names:
                    if property_name == 'KSLabel':
                        annotation_name = 'quality'
                    else:
                        annotation_name = property_name.lower()
                    for annotation_dict in annotation_list:
                        if int(annotation_dict[clust_key]) == clust_id:
                            spiketrain_an[annotation_name] = \
                                annotation_dict[property_name]
                            break

            cluster_mask = (self._spike_clusters == clust_id).flatten()

            current_templates = self._spike_templates[cluster_mask].flatten()
            unique_templates = np.unique(current_templates)
            spiketrain_an['templates'] = unique_templates
            spiketrain_an['__array_annotations__']['templates'] = current_templates

            if self._amplitudes is not None:
                spiketrain_an['__array_annotations__']['amplitudes'] = \
                    self._amplitudes[cluster_mask]

            if self._pc_features is not None:
                current_pc_features = self._pc_features[cluster_mask]
                _, num_pcs, num_pc_channels = current_pc_features.shape
                for pc_idx in range(num_pcs):
                    for channel_idx in range(num_pc_channels):
                        key = 'channel{channel_idx}_pc{pc_idx}'.format(channel_idx=channel_idx,
                                                                       pc_idx=pc_idx)
                        spiketrain_an['__array_annotations__'][key] = \
                                current_pc_features[:, pc_idx, channel_idx]

            if self._pc_feature_ind is not None:
                spiketrain_an['pc_feature_ind'] = self._pc_feature_ind[unique_templates]

    def _segment_t_start(self, block_index, seg_index):
        assert block_index == 0
        return self._t_start

    def _segment_t_stop(self, block_index, seg_index):
        assert block_index == 0
        return self._t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        return None

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return None

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                channel_indexes):
        return None

    def _spike_count(self, block_index, seg_index, spike_channel_index):
        assert block_index == 0
        spikes = self._spike_clusters
        unit_label = self.unit_labels[spike_channel_index]
        mask = spikes == unit_label
        nb_spikes = np.sum(mask)
        return nb_spikes

    def _get_spike_timestamps(self, block_index, seg_index,
                              spike_channel_index, t_start, t_stop):
        assert block_index == 0
        assert seg_index == 0

        unit_label = self.unit_labels[spike_channel_index]
        mask = self._spike_clusters == unit_label
        spike_timestamps = self._spike_times[mask]

        if t_start is not None:
            start_frame = int(t_start * self._sampling_frequency)
            spike_timestamps = \
                spike_timestamps[spike_timestamps >= start_frame]
        if t_stop is not None:
            end_frame = int(t_stop * self._sampling_frequency)
            spike_timestamps = spike_timestamps[spike_timestamps < end_frame]

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self._sampling_frequency
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index,
                                 spike_channel_index, t_start, t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):
        return None

    def _get_event_timestamps(self, block_index, seg_index,
                              event_channel_index, t_start, t_stop):
        return None

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        return None

    def _rescale_epoch_duration(self, raw_duration, dtype):
        return None

    @staticmethod
    def _parse_tsv_or_csv_to_list_of_dict(filename):
        list_of_dict = list()
        letter_pattern = re.compile('[a-zA-Z]')
        float_pattern = re.compile(r'-?\d*\.')
        with open(filename) as csvfile:
            if filename.suffix == '.csv':
                reader = csv.DictReader(csvfile, delimiter=',')
            elif filename.suffix == '.tsv':
                reader = csv.DictReader(csvfile, delimiter='\t')
            else:
                raise ValueError("Function parses only .csv or .tsv files")
            line = 0

            for row in reader:
                if line == 0:
                    cluster_id_key, *annotation_keys = tuple(row.keys())
                # Convert cluster ID to int
                row[cluster_id_key] = int(row[cluster_id_key])
                # Convert strings without letters
                for key in annotation_keys:
                    value = row[key]
                    if not len(value):
                        row[key] = None
                    elif letter_pattern.match(value) is None:
                        if float_pattern.match(value) is None:
                            row[key] = int(value)
                        else:
                            row[key] = float(value)

                list_of_dict.append(row)
                line += 1

        return list_of_dict
