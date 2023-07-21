.. _section-neo-rawio-API:

=============
Neo RawIO API
=============

.. currentmodule:: neo.rawio


For performance and memory consumption reasons, Neo provides a low-level, developer-oriented
read-only API for reading different file formats.
Neo's full-featured IO modules are built on this, but it is also available for direct use.

In brief:

- **neo.io** is the user-oriented read/write layer. Reading consists of getting a tree
  of Neo objects from a data source (file, url, or directory).
  When reading, all Neo objects are correctly scaled to the correct units.
  Writing consists of making a set of Neo objects persistent in a file format.
- **neo.rawio** is a low-level layer for reading data only. Reading consists of getting
  NumPy buffers (often int16/int64) of signals/spikes/events.
  Scaling to real values (microV, times, ...) is done in a second step.
  Here the underlying objects must be consistent across Blocks and Segments for a given
  data source.

The neo.rawio API is close in spirit to a C API for reading data, but in Python/NumPy.
Many, but not all of the file formats supported in :mod:`neo.io` also have a :mod:`neo.rawio` interface.


Possible uses of the :mod:`neo.rawio` API are:

- fast reading chunks of signals in int16 and do the scaling of units (uV)
  on a GPU while scaling the zoom. This should improve bandwidth from HD/SSD to RAM
  and from RAM to GPU memory.
- load only a small chunk of data for heavy computations. For instance
  the spike sorting module tridesclous_ does this.


The :mod:`neo.rawio` API is less flexible than :mod:`neo.io` and has some limitations:

- read-only
- AnalogSignals must have the same characteristics across all Blocks and Segments:
  ``sampling_rate``, ``shape[1]``, ``dtype``
- AnalogSignals should all have the same value of ``sampling_rate``, otherwise they won't be read
  at the same time.
- Units must have SpikeTrains even if empty across all Block and Segment
- Epoch and Event are processed the same way (with ``durations=None`` for Event).


For an intuitive comparison of :mod:`neo.io` and :mod:`neo.rawio` see:

- :doc:`examples/read_files_neo_io`
- :doc:`examples/read_files_neo_rawio`


One benefit of the :mod:`neo.rawio` API is that a developer
should be able to code a new RawIO class with little knowledge of the Neo tree of
objects or of the :mod:`quantities` package.


Basic usage
===========

.. Download example files

.. ipython:: python
   :suppress:

   import os.path
   from urllib.request import urlretrieve

   url_repo = 'https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data/raw/master/'

   for distantfile in ('plexon/File_plexon_2.plx', 'plexon/File_plexon_3.plx', 'blackrock/FileSpec2.3001.nev', 'blackrock/FileSpec2.3001.ns5'):
       localfile = distantfile.split("/")[1]
       if not os.path.exists(localfile):
          urlretrieve(url_repo + distantfile, localfile)


First create a reader from a class:

.. ipython::

    In [1]: from neo.rawio import PlexonRawIO

    In [2]: reader = PlexonRawIO(filename='File_plexon_3.plx')

Then browse the internal header and display information:

.. ipython::

    In [3]: reader.parse_header()

    In [4]: reader
    Out[4]:
    PlexonRawIO: File_plexon_3.plx
    nb_block: 1
    nb_segment:  [1]
    signal_channels: [V1]
    spike_channels: [Wspk1u, Wspk2u, Wspk4u, Wspk5u ... Wspk29u Wspk30u Wspk31u Wspk32u]
    event_channels: []

You get the number of blocks and segments per block. You have information
about channels: :attr:`signal_channels`, :attr:`spike_channels`, :attr:`event_channels`.

All this information is available in the :attr:`header` dict:

.. ipython::

    In [5]: for k, v in reader.header.items():
       ...:     print(k, v)
    Out[5]:
    signal_channels [('V1', 0,  1000., 'int16', '',  2.44140625,  0., 0)]
    event_channels []
    nb_segment [1]
    nb_block 1
    spike_channels [('Wspk1u', 'ch1#0', '',  0.00146484,  0., 0,  30000.)
    ('Wspk2u', 'ch2#0', '',  0.00146484,  0., 0,  30000.)


Read chunks of signal data and scale them
-----------------------------------------

.. ipython::

    In [6]: channel_indexes = None  #could be channel_indexes = [0]

    In [7]: raw_sigs = reader.get_analogsignal_chunk(block_index=0, seg_index=0,
       ...:                                          i_start=1024, i_stop=2048,
       ...:                                          channel_indexes=channel_indexes)

    In [8]: float_sigs = reader.rescale_signal_raw_to_float(raw_sigs, dtype='float64')

    In [9]: sampling_rate = reader.get_signal_sampling_rate()

    In [10]: t_start = reader.get_signal_t_start(block_index=0, seg_index=0)

    In [11]: units = reader.header['signal_channels'][0]['units']

    In [12]: raw_sigs.shape, raw_sigs.dtype
    Out[12]: ((1024, 1), dtype('int16'))

    In [13]: float_sigs.shape, float_sigs.dtype
    Out[13]: ((1024, 1), dtype('float64'))

    In [14]: sampling_rate, t_start, units
    Out[14]: (1000.0, 0.0, '')

There are 3 ways to select a subset of channels: by index (0 based), by id or by name.
By index is unambiguous 0 to n-1 (inclusive), whereas for some IOs channel_names
(and sometimes channel_ids) are not guaranteed to be unique.
In such cases, using names or ids may raise an error.

A selected subset of channels which is passed to :func:`get_analog_signal_chunk()`, :func:`get_analog_signal_size()`,
or :func:`get_analog_signal_t_start()` has the additional restriction that all such channels must have
the same :attr:`t_start` and :attr:`signal_size`.

Such subsets of channels may be available in specific RawIOs by using the
:func:`get_group_signal_channel_indexes()` method, if the RawIO has defined separate
:attr:`group_ids` for each group with those common characteristics.

Example with BlackrockRawIO for the recording `FileSpec2.3001`_:

.. ipython::

    In [15]: from neo.rawio import BlackrockRawIO

    In [16]: reader = BlackrockRawIO(filename="FileSpec2.3001")

    In [17]: reader.parse_header()

    In [18]: raw_sigs = reader.get_analogsignal_chunk(channel_indexes=None)  # Take all channels

    In [19]: raw_sigs1 = reader.get_analogsignal_chunk(channel_indexes=[0, 2, 4])  # Take 0 2 and 4

    In [20]: raw_sigs2 = reader.get_analogsignal_chunk(channel_ids=['1', '3', '5'])  # Same but with their id (1 based)

    In [21]: raw_sigs3 = reader.get_analogsignal_chunk(channel_names=['chan1', 'chan3', 'chan5'])  # Same but with their name

    In [22]: raw_sigs1.shape[1], raw_sigs2.shape[1], raw_sigs3.shape[1]
    Out[22]: (3, 3, 3)



Inspect spiking unit channels
-----------------------------

Each channel gives a SpikeTrain for each Segment.
Note that for many formats a physical channel can have several units after spike
sorting. So the number of spike channels could be more than the number of physical channels or signal channels.

.. ipython::

    In [23]: nb_unit = reader.spike_channels_count()

    In [24]: print('nb_unit', nb_unit)
    nb_unit 4

    In [25]: for spike_channel_index in range(nb_unit):
       ....:     nb_spike = reader.spike_count(block_index=0, seg_index=0, spike_channel_index=spike_channel_index)
       ....:     print('spike_channel_index', spike_channel_index, 'nb_spike', nb_spike)
    spike_channel_index 0 nb_spike 259
    spike_channel_index 1 nb_spike 234
    spike_channel_index 2 nb_spike 218
    spike_channel_index 3 nb_spike 253


Get spike timestamps in a defined time range and convert them to spike times
----------------------------------------------------------------------------

.. ipython::

    In [26]: spike_timestamps = reader.get_spike_timestamps(block_index=0, seg_index=0, spike_channel_index=0,
       ....:                                                t_start=0, t_stop=10)

    In [27]: print(spike_timestamps.shape, spike_timestamps.dtype, spike_timestamps[:5])
    (86,) uint32 [ 19312  49298  79301 139290 162170]

    In [28]: spike_times =  reader.rescale_spike_timestamp( spike_timestamps, dtype='float64')

    In [29]: print(spike_times.shape, spike_times.dtype, spike_times[:5])
    (86,) float64 [0.64373333 1.64326667 2.64336667 4.643      5.40566667]


Get spike waveforms in a defined time range
-------------------------------------------

.. ipython::

    In [30]: raw_waveforms = reader.get_spike_raw_waveforms(block_index=0, seg_index=0, spike_channel_index=0,
       ....:                                                t_start=0, t_stop=10)

    In [31]: print(raw_waveforms.shape, raw_waveforms.dtype, raw_waveforms[0, 0, :4])
    (86, 1, 48) int16 [-209 -224  -74  205]

    In [32]: float_waveforms = reader.rescale_waveforms_to_float(raw_waveforms, dtype='float32', spike_channel_index=0)

    In [33]: print(float_waveforms.shape, float_waveforms.dtype, float_waveforms[0,0,:4])
    (86, 1, 48) float32 [-52.25 -56.   -18.5   51.25]


Count events per channel
------------------------

.. ipython::

    In [34]: reader = PlexonRawIO(filename='File_plexon_2.plx')

    In [35]: reader.parse_header()

    In [36]: nb_event_channel = reader.event_channels_count()

    In [37]: print('nb_event_channel', nb_event_channel)
    nb_event_channel 28

    In [38]: for chan_index in range(nb_event_channel):
       ....:     nb_event = reader.event_count(block_index=0, seg_index=0, event_channel_index=chan_index)
       ....:     print('chan_index',chan_index, 'nb_event', nb_event)
    chan_index 0 nb_event 1
    chan_index 1 nb_event 0
    chan_index 2 nb_event 0
    chan_index 3 nb_event 0
    ...



Read event timestamps and times
-------------------------------

.. ipython::

    In [39]: ev_timestamps, ev_durations, ev_labels = reader.get_event_timestamps(
       ....:    block_index=0, seg_index=0, event_channel_index=0,
       ....:    t_start=None, t_stop=None)

    In [40]: print(ev_timestamps, ev_durations, ev_labels)
    [1268] None ['0']

    In [41]: ev_times = reader.rescale_event_timestamp(ev_timestamps, dtype='float64')

    In [42]: print(ev_times)
    [ 0.0317]




List of implemented formats
===========================

See :doc:`rawiolist`.


.. _tridesclous: https://github.com/tridesclous/tridesclous
.. _FileSpec2.3001: https://gin.g-node.org/NeuralEnsemble/ephy_testing_data/src/master/blackrock