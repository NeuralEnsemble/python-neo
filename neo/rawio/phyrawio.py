"""
PhyRawIO is a class to handle Phy spike sorting data.
Ported from:
https://github.com/SpikeInterface/spikeextractors/blob/
f20b1219eba9d3330d5d7cd7ce8d8924a255b8c2/spikeextractors/
extractors/phyextractors/phyextractors.py

ExampleRawIO is a class of a  fake example.
This is to be used when coding a new RawIO.


Rules for creating a new class:
  1. Step 1: Create the main class
    * Create a file in **neo/rawio/** that endith with "rawio.py"
    * Create the class that inherits BaseRawIO
    * copy/paste all methods that need to be implemented.
      See the end a neo.rawio.baserawio.BaseRawIO
    * code hard! The main difficulty **is _parse_header()**.
      In short you have a create a mandatory dict than
      contains channel informations::

            self.header = {}
            self.header['nb_block'] = 2
            self.header['nb_segment'] = [2, 3]
            self.header['signal_channels'] = sig_channels
            self.header['unit_channels'] = unit_channels
            self.header['event_channels'] = event_channels

  2. Step 2: RawIO test:
    * create a file in neo/rawio/tests with the same name with "test_" prefix
    * copy paste neo/rawio/tests/test_examplerawio.py and do the same

  3. Step 3 : Create the neo.io class with the wrapper
    * Create a file in neo/io/ that ends with "io.py"
    * Create a class that inherits both your RawIO class and BaseFromRaw class
    * copy/paste from neo/io/exampleio.py

  4.Step 4 : IO test
    * create a file in neo/test/iotest with the same previous name with "test_" prefix
    * copy/paste from neo/test/iotest/test_exampleio.py



"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np
from pathlib import Path
import re
import csv
import ast


class PhyRawIO(BaseRawIO):
    """
    Class for "reading" fake data from an imaginary file.

    For the user, it gives access to raw data (signals, event, spikes) as they
    are in the (fake) file int16 and int64.

    For a developer, it is just an example showing guidelines for someone who wants
    to develop a new IO module.

    Two rules for developers:
      * Respect the :ref:`neo_rawio_API`
      * Follow the :ref:`io_guiline`

    This fake IO:
        * has 2 blocks
        * blocks have 2 and 3 segments
        * has 16 signal_channels sample_rate = 10000
        * has 3 unit_channels
        * has 2 event channels: one has *type=event*, the other has
          *type=epoch*


    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.PhyRawIO(dirname='/dir/to/phy/folder')
        >>> r.parse_header()
        >>> print(r)
        >>> spike_timestamp = r.get_spike_timestamps(block_index=0,
        ... seg_index=0, unit_index=0, t_start=None, t_stop=None)
        >>> spike_times = r.rescale_spike_timestamp(spike_timestamp, 'float64')

    """
    extensions = []
    rawmode = 'one-folder'

    def __init__(self, dirname=''):
        BaseRawIO.__init__(self)
        self.dirname = dirname

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

        # TODO: Add this when array_annotations are ready
        # if (phy_folder / 'amplitudes.npy').is_file():
        #     amplitudes = np.squeeze(np.load(phy_folder / 'amplitudes.npy'))
        # else:
        #     amplitudes = np.ones(len(spike_times))
        #
        # if (phy_folder / 'pc_features.npy').is_file():
        #     pc_features = np.squeeze(np.load(phy_folder / 'pc_features.npy'))
        # else:
        #     pc_features = None

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

        sig_channels = []
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        unit_channels = []
        for i, clust_id in enumerate(clust_ids):
            unit_name = f'unit {clust_id}'
            unit_id = f'{clust_id}'
            wf_units = ''
            wf_gain = 0
            wf_offset = 0.
            wf_left_sweep = 0
            wf_sampling_rate = 0
            unit_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                  wf_offset, wf_left_sweep, wf_sampling_rate))
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)

        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

        csv_tsv_files = [x for x in phy_folder.iterdir() if
                         x.suffix == '.csv' or x.suffix == '.tsv']

        # annotation_lists is list of list of dict (python==3.8)
        # or list of list of ordered dict (python==3.6)
        # SEE: https://docs.python.org/3/library/csv.html#csv.DictReader
        annotation_lists = [self._parse_tsv_or_csv_to_list_of_dict(file)
                            for file in csv_tsv_files]

        for block_index in range(1):
            bl_ann = self.raw_annotations['blocks'][block_index]
            bl_ann['name'] = f'Block #{block_index}'
            bl_ann['block_extra_info'] = f'This is the block {block_index}'
            for seg_index in range([1][block_index]):
                seg_ann = bl_ann['segments'][seg_index]
                seg_ann['name'] = f'Seg #{seg_index} Block #{block_index}'
                seg_ann['seg_extra_info'] = f'This is the seg {seg_index} of ' \
                                            f'block {block_index}'
                for index, clust_id in enumerate(clust_ids):
                    spiketrain_an = seg_ann['units'][index]

                    # Loop over list of list of dict and annotate each st
                    for annotation_list in annotation_lists:
                        clust_key, property_name = tuple(annotation_list[0].keys())
                        if property_name == 'KSLabel':
                            annotation_name = 'quality'
                        else:
                            annotation_name = property_name.lower()
                        for annotation_dict in annotation_list:
                            if int(annotation_dict[clust_key]) == clust_id:
                                spiketrain_an[annotation_name] = annotation_dict[property_name]
                                break

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        pass

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        return None

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return None

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        return None

    def _spike_count(self, block_index, seg_index, unit_index):
        nb_spikes = len(self._spike_times)
        return nb_spikes

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        assert block_index == 0
        assert seg_index == 0

        unit_label = self.unit_labels[unit_index]
        mask = self._spike_clusters == unit_label
        spike_timestamps = self._spike_times[mask]

        if t_start is not None:
            start_frame = int(t_start * self._sampling_frequency)
            spike_timestamps = spike_timestamps[spike_timestamps >= start_frame]
        if t_stop is not None:
            end_frame = int(t_stop * self._sampling_frequency)
            spike_timestamps = spike_timestamps[spike_timestamps < end_frame]

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self._sampling_frequency
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):
        return None

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        return None

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        return None

    def _rescale_epoch_duration(self, raw_duration, dtype):
        return None

    @staticmethod
    def _parse_tsv_or_csv_to_list_of_dict(filename):
        list_of_dict = list()
        with open(filename) as csvfile:
            if filename.suffix == '.csv':
                reader = csv.DictReader(csvfile, delimiter=',')
            elif filename.suffix == '.tsv':
                reader = csv.DictReader(csvfile, delimiter='\t')
            else:
                raise ValueError("Function parses only .csv or .tsv files")
            for row in reader:
                list_of_dict.append(row)

        return list_of_dict

