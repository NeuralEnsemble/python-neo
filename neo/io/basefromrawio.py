"""
BaseFromRaw
======

BaseFromRaw implement a bridge between the new neo.rawio API
and the neo.io legacy that give neo.core object.
The neo.rawio API is more restricted and limited and does not cover tricky
cases with asymetrical tree of neo object.
But if a format is done in neo.rawio the neo.io is done for free
by inheritance of this class.
Furthermore, IOs that inherit this BaseFromRaw also have the ability
to lazy load with proxy objects.


"""

import collections
import warnings
import numpy as np

from neo import logging_handler
from neo.core import AnalogSignal, Block, Epoch, Event, IrregularlySampledSignal, Group, Segment, SpikeTrain
from neo.io.baseio import BaseIO

from neo.io.proxyobjects import (
    AnalogSignalProxy,
    SpikeTrainProxy,
    EventProxy,
    EpochProxy,
    ensure_signal_units,
    check_annotations,
    ensure_second,
    proxyobjectlist,
)


import quantities as pq


class BaseFromRaw(BaseIO):
    """
    This implement generic reader on top of RawIO reader.

    Arguments depend on `mode` (dir or file)

    File case::

        reader = BlackRockIO(filename='FileSpec2.3001.nev')

    Dir case::

        reader = NeuralynxIO(dirname='Cheetah_v5.7.4/original_data')

    Other arguments are IO specific.

    """

    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal, SpikeTrain, Group, Event, Epoch]
    readable_objects = [Block, Segment]
    writeable_objects = []

    support_lazy = True

    name = "BaseIO"
    description = ""
    extensions = []

    mode = "file"

    _prefered_signal_group_mode = "group-by-same-units"  # 'split-all'

    def __init__(self, *args, **kargs):
        BaseIO.__init__(self, *args, **kargs)
        self.parse_header()

    def read_block(
        self, block_index=0, lazy=False, create_group_across_segment=None, signal_group_mode=None, load_waveforms=False
    ):
        """
        Reads one block of data from a file

        Parameters
        ----------
        block_index: int, default: 0
            In the case of multiple blocks, the block_index specifies which block to read
        lazy: bool, default: False
            Whether to read the block lazily (True) or load into memory (False)
        create_group_across_segment: bool | dict | None, default: None
            If True :
                * Create a neo.Group to group AnalogSignal segments
                * Create a neo.Group to group SpikeTrain across segments
                * Create a neo.Group to group Event across segments
                * Create a neo.Group to group Epoch across segments
            With a dict the behavior can be controlled more finely
                * for example: create_group_across_segment = { 'AnalogSignal': True, 'SpikeTrain': False, ...}
        signal_group_mode: 'split-all' | 'group-by-same-units' | None, default: None
            This control behavior for grouping channels in AnalogSignal.
                * 'split-all': each channel will be give an AnalogSignal
                * 'group-by-same-units' all channel sharing the same quantity units are grouped in a 2D AnalogSignal
            By default None since the default is dependant on the IO
        load_waveforms: bool, default: False
            Determines whether SpikeTrains.waveforms is created

        Returns
        -------
        bl: neo.core.Block
            The block of data

        """

        if signal_group_mode is None:
            signal_group_mode = self._prefered_signal_group_mode

        l = ["AnalogSignal", "SpikeTrain", "Event", "Epoch"]
        if create_group_across_segment is None:
            # @andrew @ julia @michael ?
            # I think here the default None could give this
            create_group_across_segment = {
                "AnalogSignal": True,  # because mimic the old ChannelIndex for AnalogSignals
                "SpikeTrain": False,  # False by default because can create too many object for simulation
                "Event": False,  # not implemented yet
                "Epoch": False,  # not implemented yet
            }
        elif isinstance(create_group_across_segment, bool):
            # bool to dict
            v = create_group_across_segment
            create_group_across_segment = {k: v for k in l}
        elif isinstance(create_group_across_segment, dict):
            # put False to missing keys
            create_group_across_segment = {k: create_group_across_segment.get(k, False) for k in l}
        else:
            raise ValueError("create_group_across_segment must be bool or dict")

        # annotations
        bl_annotations = dict(self.raw_annotations["blocks"][block_index])
        bl_annotations.pop("segments")
        bl_annotations = check_annotations(bl_annotations)

        bl = Block(**bl_annotations)

        # Group for AnalogSignals coming from signal_streams
        if create_group_across_segment["AnalogSignal"]:
            signal_streams = self.header["signal_streams"]
            sub_streams = self.get_sub_signal_streams(signal_group_mode)
            sub_stream_groups = []
            for sub_stream in sub_streams:
                stream_index, inner_stream_channels, name = sub_stream
                group = Group(name=name, stream_id=signal_streams[stream_index]["id"])
                bl.groups.append(group)
                sub_stream_groups.append(group)

        if create_group_across_segment["SpikeTrain"]:
            spike_channels = self.header["spike_channels"]
            st_groups = []
            for c in range(spike_channels.size):
                group = Group(name=f"SpikeTrain group {c}")
                group.annotate(unit_name=spike_channels[c]["name"])
                group.annotate(unit_id=spike_channels[c]["id"])
                bl.groups.append(group)
                st_groups.append(group)

        if create_group_across_segment["Event"]:
            # @andrew @ julia @michael :
            # Do we need this ? I guess yes
            raise NotImplementedError()

        if create_group_across_segment["Epoch"]:
            # @andrew @ julia @michael :
            # Do we need this ? I guess yes
            raise NotImplementedError()

        # Read all segments
        for seg_index in range(self.segment_count(block_index)):
            seg = self.read_segment(
                block_index=block_index,
                seg_index=seg_index,
                lazy=lazy,
                signal_group_mode=signal_group_mode,
                load_waveforms=load_waveforms,
            )
            bl.segments.append(seg)

        # create link between group (across segment) and data objects
        for seg in bl.segments:
            if create_group_across_segment["AnalogSignal"]:
                for c, anasig in enumerate(seg.analogsignals):
                    sub_stream_groups[c].add(anasig)

            if create_group_across_segment["SpikeTrain"]:
                for c, sptr in enumerate(seg.spiketrains):
                    st_groups[c].add(sptr)

        bl.check_relationships()

        return bl

    def read_segment(
        self,
        block_index=0,
        seg_index=0,
        lazy=False,
        signal_group_mode=None,
        load_waveforms=False,
        time_slice=None,
        strict_slicing=True,
    ):
        """
        Reads one segment of the data file

        Parameters
        ----------
        block_index: int, default: 0
            In the case of a multiblock dataset, this specifies which block to read
        seg_index: int, default: 0
            In the case of a multisegment block, this specifies which segmeent to read
        lazy: bool, default: False
            Whether to lazily load the segment (True) or to load the segment into memory (False)
        signal_group_mode: 'split-all' | 'group-by-same-units' | None, default: None
           This control behavior for grouping channels in AnalogSignal.
            * 'split-all': each channel will be give an AnalogSignal
            * 'group-by-same-units' all channel sharing the same quantity units are grouped in a 2D AnalogSignal
        load_waveforms: bool, default: False
            Determines whether SpikeTrains.waveforms is created
        time_slice: tuple[quantity.Quantities | None] | None, default: None
            Whether to take a time slice of the data
                * None: indicates from beginning of the segment to the end of the segment
                * tuple: (t_start, t_stop) with t_start and t_stop being quantities in seconds
                * tuple: (None, t_stop) indicates the beginning of the segment to t_stop
                * tuple: (t_start, None) indicates from t_start to the end of the segment
        strict_slicing: bool, default: True
            Control if an error is raised or not when t_start or t_stop
            is outside of the real time range of the segment.

        Returns
        -------
        seg: neo.core.Segment
            Returns the desired segment based on the parameters given
        """

        if lazy:
            if time_slice is not None:
                raise ValueError("For lazy=True you must specify a time_slice when LazyObject.load(time_slice=...)")
            if load_waveforms:
                raise ValueError(
                    "For lazy=True you must specify load_waveforms when SpikeTrain.load(load_waveforms=...)"
                )

        if signal_group_mode is None:
            signal_group_mode = self._prefered_signal_group_mode

        # annotations
        seg_annotations = self.raw_annotations["blocks"][block_index]["segments"][seg_index].copy()
        for k in ("signals", "spikes", "events"):
            seg_annotations.pop(k)
        seg_annotations = check_annotations(seg_annotations)

        seg = Segment(index=seg_index, **seg_annotations)

        # AnalogSignal
        signal_streams = self.header["signal_streams"]
        sub_streams = self.get_sub_signal_streams(signal_group_mode)
        for sub_stream in sub_streams:
            stream_index, inner_stream_channels, name = sub_stream
            anasig = AnalogSignalProxy(
                rawio=self,
                stream_index=stream_index,
                inner_stream_channels=inner_stream_channels,
                block_index=block_index,
                seg_index=seg_index,
            )
            anasig.name = name

            if not lazy:
                # ... and get the real AnalogSignal if not lazy
                anasig = anasig.load(time_slice=time_slice, strict_slicing=strict_slicing)

            seg.analogsignals.append(anasig)

        # SpikeTrain and waveforms (optional)
        spike_channels = self.header["spike_channels"]
        for spike_channel_index in range(len(spike_channels)):
            # make a proxy...
            sptr = SpikeTrainProxy(
                rawio=self, spike_channel_index=spike_channel_index, block_index=block_index, seg_index=seg_index
            )

            if not lazy:
                # ... and get the real SpikeTrain if not lazy
                sptr = sptr.load(time_slice=time_slice, strict_slicing=strict_slicing, load_waveforms=load_waveforms)
                # TODO magnitude_mode='rescaled'/'raw'

            seg.spiketrains.append(sptr)

        # Events/Epoch
        event_channels = self.header["event_channels"]
        for chan_ind in range(len(event_channels)):
            if event_channels["type"][chan_ind] == b"event":
                e = EventProxy(rawio=self, event_channel_index=chan_ind, block_index=block_index, seg_index=seg_index)
                if not lazy:
                    e = e.load(time_slice=time_slice, strict_slicing=strict_slicing)
                seg.events.append(e)
            elif event_channels["type"][chan_ind] == b"epoch":
                e = EpochProxy(rawio=self, event_channel_index=chan_ind, block_index=block_index, seg_index=seg_index)
                if not lazy:
                    e = e.load(time_slice=time_slice, strict_slicing=strict_slicing)
                seg.epochs.append(e)

        seg.check_relationships()
        return seg

    def get_sub_signal_streams(self, signal_group_mode="group-by-same-units"):
        """
        When signal streams don't have homogeneous SI units across channels,
        they have to be split in sub streams to construct AnalogSignal objects with unique units.

        For backward compatibility (neo version <= 0.5) sub-streams can also be
        used to generate one AnalogSignal per channel.
        """
        signal_streams = self.header["signal_streams"]
        signal_channels = self.header["signal_channels"]

        sub_streams = []
        for stream_index in range(len(signal_streams)):
            stream_id = signal_streams[stream_index]["id"]
            stream_name = signal_streams[stream_index]["name"]
            mask = signal_channels["stream_id"] == stream_id
            channels = signal_channels[mask]
            if signal_group_mode == "group-by-same-units":
                # this does not keep the original order
                _, idx = np.unique(channels["units"], return_index=True)
                all_units = channels["units"][np.sort(idx)]

                if len(all_units) == 1:
                    # no substream
                    # None iwill be transform as slice later
                    inner_stream_channels = None
                    name = stream_name
                    sub_stream = (stream_index, inner_stream_channels, name)
                    sub_streams.append(sub_stream)
                else:
                    for units in all_units:
                        (inner_stream_channels,) = np.nonzero(channels["units"] == units)
                        chan_names = channels[inner_stream_channels]["name"]
                        name = "Channels: (" + " ".join(chan_names) + ")"
                        sub_stream = (stream_index, inner_stream_channels, name)
                        sub_streams.append(sub_stream)
            elif signal_group_mode == "split-all":
                # mimic all neo <= 0.5 behavior
                for i, channel in enumerate(channels):
                    inner_stream_channels = [i]
                    name = channels[i]["name"]
                    sub_stream = (stream_index, inner_stream_channels, name)
                    sub_streams.append(sub_stream)
            else:
                raise (NotImplementedError)

        return sub_streams
