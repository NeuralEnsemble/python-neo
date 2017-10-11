# -*- coding: utf-8 -*-
"""
This is an example for reading files with neo.rawio
compare with read_files_neo_io.py
"""

import urllib
from neo.rawio import PlexonRawIO



#Get Plexon files
distantfile = 'https://portal.g-node.org/neo/plexon/File_plexon_3.plx'
localfile = './File_plexon_3.plx'
urllib.request.urlretrieve(distantfile, localfile)


# create a reader
reader = PlexonRawIO(filename='File_plexon_3.plx')
reader.parse_header()
print(reader)
print(reader.header)

#Read signal chunks
channel_indexes = None #could be channel_indexes = [0]
raw_sigs = reader.get_analogsignal_chunk(block_index=0, seg_index=0, 
                        i_start=1024, i_stop=2048, channel_indexes=channel_indexes)
float_sigs = reader.rescale_signal_raw_to_float(raw_sigs, dtype='float64')
sampling_rate = reader.get_signal_sampling_rate()
t_start = reader.get_signal_t_start(block_index=0, seg_index=0)
units =reader.header['signal_channels'][0]['units']
print(raw_sigs.shape, raw_sigs.dtype)
print(float_sigs.shape, float_sigs.dtype)
print(sampling_rate, t_start, units)


#Count unit and spike per units
nb_unit = reader.unit_channels_count()
print('nb_unit', nb_unit)
for unit_index in range(nb_unit):
    nb_spike = reader.spike_count(block_index=0, seg_index=0, unit_index=unit_index)
    print('unit_index', unit_index, 'nb_spike', nb_spike)


#Read spike times
spike_timestamps = reader.spike_timestamps(block_index=0, seg_index=0, unit_index=0,
                        t_start=0., t_stop=10.)
print(spike_timestamps.shape, spike_timestamps.dtype, spike_timestamps[:5])
spike_times =  reader.rescale_spike_timestamp( spike_timestamps, dtype='float64')
print(spike_times.shape, spike_times.dtype, spike_times[:5])


#Read spike waveforms
raw_waveforms = reader.spike_raw_waveforms(  block_index=0, seg_index=0, unit_index=0,
                        t_start=0., t_stop=10.)
print(raw_waveforms.shape, raw_waveforms.dtype, raw_waveforms[0,0,:4])
float_waveforms = reader.rescale_waveforms_to_float(raw_waveforms, dtype='float32', unit_index=0)
print(float_waveforms.shape, float_waveforms.dtype, float_waveforms[0,0,:4])


#Read event timestamps and times (take anotehr file)
distantfile = 'https://portal.g-node.org/neo/plexon/File_plexon_2.plx'
localfile = './File_plexon_2.plx'
urllib.request.urlretrieve(distantfile, localfile)


#Count event per channel
reader = PlexonRawIO(filename='File_plexon_2.plx')
reader.parse_header()
nb_event_channel = reader.event_channels_count()
print('nb_event_channel', nb_event_channel)
for chan_index in range(nb_event_channel):
    nb_event = reader.event_count(block_index=0, seg_index=0, event_channel_index=chan_index)
    print('chan_index',chan_index, 'nb_event', nb_event)


ev_timestamps, ev_durations, ev_labels = reader.event_timestamps(block_index=0, seg_index=0, event_channel_index=0,
                        t_start=None, t_stop=None)
print(ev_timestamps, ev_durations, ev_labels)
ev_times = reader.rescale_event_timestamp(ev_timestamps, dtype='float64')
print(ev_times)
