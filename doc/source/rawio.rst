*********
Neo RawIO
*********

.. currentmodule:: neo.rawio


.. _neo_rawio_API:


For performance and memory consumption reasons a new layer has been added to Neo.

In brief:
    * **neo.io** is the user-oriented read/write layer. Reading consists of getting a tree
      of Neo objects from a data source (file, url, or directory). 
      When  reading, all Neo objects are correctly scaled to the correct units.
      Writing consists of making a set of Neo objects persistent in a file format.
    * **neo.rawio** is a low-level layer for reading data only. Reading consists of getting
      NumPy buffers (often int16/int64) of signals/spikes/events.
      Scaling to real values (microV, times, ...) is done in a second step.
      Here the underlying objects must be consistent across Blocks and Segments for a given
      data source.

      
The neo.rawio API has been added for developers.
The neo.rawio is close to what could be a C API for reading data but in Python/NumPy.


Not all IOs are implemented in :mod:`neo.rawio` but all classes implemented in :mod:`neo.rawio` are
also available in :mod:`neo.io`.


Possible uses of the :mod:`neo.rawio` API are:
    * fast reading chunks of signals in int16 and do the scaling of units (uV)
      on a GPU while scaling the zoom. This should improve bandwith HD to RAM
      and RAM to GPU memory.
    * load only some small chunk of data for heavy computations. For instance
      the spike sorting module tridesclous_ does this.


The :mod:`neo.rawio` API is less flexible than :mod:`neo.io` and has some limitations:
  * read-only
  * AnalogSignals must have the same characteristcs across all Blocks and Segments:
    ``sampling_rate``, ``shape[1]``, ``dtype``
  * AnalogSignals should all have the same value of ``sampling_rate``, otherwise they won't be read
    at the same time.
  * Units must have SpikeTrain event if empty across all Block and Segment
  * Epoch and Event are processed the same way (with ``durations=None`` for Event).

    
For an intuitive comparison of :mod:`neo.io` and :mod:`neo.rawio` see:
  * :file:`example/read_file_neo_io.py`
  * :file:`example/read_file_neo_rawio.py`

  
One speculative benefit of the :mod:`neo.rawio` API should be that a developer 
should be able to code a new RawIO class with little knowledge of the Neo tree of 
objects or of the :mod:`quantities` package.


Basic usage
===========


First create a reader from a class::

    >>> from neo.rawio import PlexonRawIO
    >>> reader = PlexonRawIO(filename='File_plexon_3.plx')

Then browse the internal header and display information::

    >>> reader.parse_header()
    >>> print(reader)
    PlexonRawIO: File_plexon_3.plx
    nb_block: 1
    nb_segment:  [1]
    signal_channels: [V1]
    unit_channels: [Wspk1u, Wspk2u, Wspk4u, Wspk5u ... Wspk29u Wspk30u Wspk31u Wspk32u]
    event_channels: []

You get the number of blocks and segments per block. You have information
about channels: **signal_channels**, **unit_channels**, **event_channels**.

All this information is internally available in the *header* dict::

    >>> for k, v in reader.header.items():
    ...    print(k, v)
    signal_channels [('V1', 0,  1000., 'int16', '',  2.44140625,  0., 0)]
    event_channels []
    nb_segment [1]
    nb_block 1
    unit_channels [('Wspk1u', 'ch1#0', '',  0.00146484,  0., 0,  30000.)
    ('Wspk2u', 'ch2#0', '',  0.00146484,  0., 0,  30000.)
    ...


Read signal chunks of data and scale them::

    >>> channel_indexes = None  #could be channel_indexes = [0]
    >>> raw_sigs = reader.get_analogsignal_chunk(block_index=0, seg_index=0, 
                        i_start=1024, i_stop=2048, channel_indexes=channel_indexes)
    >>> float_sigs = reader.rescale_signal_raw_to_float(raw_sigs, dtype='float64')
    >>> sampling_rate = reader.get_signal_sampling_rate()
    >>> t_start = reader.get_signal_t_start(block_index=0, seg_index=0)
    >>> units =reader.header['signal_channels'][0]['units']
    >>> print(raw_sigs.shape, raw_sigs.dtype)
    >>> print(float_sigs.shape, float_sigs.dtype)
    >>> print(sampling_rate, t_start, units)
    (1024, 1) int16
    (1024, 1) float64
    1000.0 0.0 V


There are 3 ways to select a subset of channels: by index (0 based), by id or by name.
By index is not ambiguous 0 to n-1 (included), for some IOs channel_names (and sometimes channel_ids) have no guarantees to
be unique, in such cases it would raise an error.

Example with BlackrockRawIO for the file FileSpec2.3001::

    >>> raw_sigs = reader.get_analogsignal_chunk(channel_indexes=None) #Take all channels
    >>> raw_sigs1 = reader.get_analogsignal_chunk(channel_indexes=[0,  2, 4])) #Take 0 2 and 4
    >>> raw_sigs2 = reader.get_analogsignal_chunk(channel_ids=[1, 3, 5]) # Same but with there id (1 based)
    >>> raw_sigs3 = reader.get_analogsignal_chunk(channel_names=['chan1', 'chan3', 'chan5'])) # Same but with there name
    print(raw_sigs1.shape[1], raw_sigs2.shape[1], raw_sigs3.shape[1])
    3, 3, 3



Inspect units channel. Each channel gives a SpikeTrain for each Segment.
Note that for many formats a physical channel can have several units after spike
sorting. So the nb_unit could be more than physical channel or signal channels.

    >>> nb_unit = reader.unit_channels_count()
    >>> print('nb_unit', nb_unit)
    nb_unit 30
    >>> for unit_index in range(nb_unit):
    ...     nb_spike = reader.spike_count(block_index=0, seg_index=0, unit_index=unit_index)
    ...     print('unit_index', unit_index, 'nb_spike', nb_spike)
    unit_index 0 nb_spike 701
    unit_index 1 nb_spike 716
    unit_index 2 nb_spike 69
    unit_index 3 nb_spike 12
    unit_index 4 nb_spike 95
    unit_index 5 nb_spike 37
    unit_index 6 nb_spike 25
    unit_index 7 nb_spike 15
    unit_index 8 nb_spike 33
    ...

    
Get spike timestamps only between 0 and 10 seconds and convert them to spike times::

    >>> spike_timestamps = reader.spike_timestamps(block_index=0, seg_index=0, unit_index=0,
                        t_start=0., t_stop=10.)
    >>> print(spike_timestamps.shape, spike_timestamps.dtype, spike_timestamps[:5])
    (424,) int64 [  90  420  708 1020 1310]
    >>> spike_times =  reader.rescale_spike_timestamp( spike_timestamps, dtype='float64')
    >>> print(spike_times.shape, spike_times.dtype, spike_times[:5])
    (424,) float64 [ 0.003       0.014       0.0236      0.034       0.04366667]


Get spike waveforms between 0 and 10 s::

    >>> raw_waveforms = reader.spike_raw_waveforms(  block_index=0, seg_index=0, unit_index=0,
                        t_start=0., t_stop=10.)
    >>> print(raw_waveforms.shape, raw_waveforms.dtype, raw_waveforms[0,0,:4])
    (424, 1, 64) int16 [-449 -206   34   40]
    >>> float_waveforms = reader.rescale_waveforms_to_float(raw_waveforms, dtype='float32', unit_index=0)
    >>> print(float_waveforms.shape, float_waveforms.dtype, float_waveforms[0,0,:4])
    (424, 1, 64) float32 [-0.65771484 -0.30175781  0.04980469  0.05859375]



Count events per channel::

    >>> reader = PlexonRawIO(filename='File_plexon_2.plx')
    >>> reader.parse_header()
    >>> nb_event_channel = reader.event_channels_count()
    nb_event_channel 28
    >>> print('nb_event_channel', nb_event_channel)
    >>> for chan_index in range(nb_event_channel):
    ...     nb_event = reader.event_count(block_index=0, seg_index=0, event_channel_index=chan_index)
    ...     print('chan_index',chan_index, 'nb_event', nb_event)
    chan_index 0 nb_event 1
    chan_index 1 nb_event 0
    chan_index 2 nb_event 0
    chan_index 3 nb_event 0
    ...

   

Read event timestamps and times for chanindex=0 and with time limits (t_start=None, t_stop=None)::

    >>> ev_timestamps, ev_durations, ev_labels = reader.event_timestamps(block_index=0, seg_index=0, event_channel_index=0,
                        t_start=None, t_stop=None)
    >>> print(ev_timestamps, ev_durations, ev_labels)
    [1268] None ['0']
    >>> ev_times = reader.rescale_event_timestamp(ev_timestamps, dtype='float64')
    >>> print(ev_times)
    [ 0.0317]




.. _tridesclous: https://github.com/tridesclous/tridesclous
