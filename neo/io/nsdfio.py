# -*- coding: utf-8 -*-

"""
Module for reading and writing NSDF files

Author: Mieszko Grodzicki

This module support both reading and writing NDSF files.
Note: Read file must be written using this IO
"""

from __future__ import absolute_import

import numpy as np
import quantities as pq

from uuid import uuid1
import pickle
from datetime import datetime
import os

try:
    import nsdf
except ImportError as err:
    HAVE_NSDF = False
    NSDF_ERR = err
else:
    HAVE_NSDF = True
    NSDF_ERR = None

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal


class NSDFIO(BaseIO):
    """
    Class for reading and writing files in NSDF Format.

    It supports reading and writing:
    - :class:'Block'
    - :class:'Segment'
    - :class:'AnalogSignal'

    with all relationships and metadata.
    """
    is_readable = True
    is_writable = True

    supported_objects = [Block, Segment, AnalogSignal]

    readable_objects = [Block, Segment]
    writeable_objects = [Block, Segment]

    has_header = False
    is_streameable = False

    name = 'NSDF'
    extensions = ['h5']
    mode = 'file'

    def __init__(self, filename=None):
        """
        Initialise NSDFIO instance

        :param filename: Path to the file
        """
        if not HAVE_NSDF:
            raise Exception("Failed to import NSDF.")

        if filename is None:
            raise ValueError("Must provide an input file.")

        BaseIO.__init__(self)

        self.filename = filename
        self.dt_format = '%d/%m/%Y %H:%M:%S'
        self.modeltree_path = '/model/modeltree/neo/'

    def write_all_blocks(self, blocks):
        """
        Write list of blocks to the file

        :param blocks: List of blocks to be written
        """
        writer = self._init_writing()
        neo_model, blocks_model, segments_model = self._prepare_model_tree()

        name_pattern = self._name_pattern(len(blocks))
        for i, block in enumerate(blocks):
            self.write_block(block, name_pattern.format(i), writer, blocks_model)

        writer.add_modeltree(neo_model)

    def write_block(self, block = None, name='0', writer=None, parent=None):
        """
        Write a Block to the file

        :param block: Block to be written
        :param name: Name for block representation in NSDF model tree (optional)
        :param writer: NSDFWriter instance (optional)
        :param parent: NSDF ModelComponent which will be the parent of block NSDF representation (optional)
        """
        if not isinstance(block, Block):
            raise ValueError("Must provide a Block to write.")

        if writer is None:
            writer = self._init_writing()

        single_block = False
        if parent is None:
            neo_model, parent, segments_model = self._prepare_model_tree()
            single_block = True

        block_model = nsdf.ModelComponent(name, uid=uuid1().hex, parent=parent)

        self._write_block_children(block, block_model, writer)
        self._write_container_metadata(block, block_model)

        if single_block:
            writer.add_modeltree(neo_model)

    def _write_block_children(self, block, block_model, writer):
        segments_model = nsdf.ModelComponent(name='segments', uid=uuid1().hex, parent=block_model)
        name_pattern = '{{:0{}d}}'.format(self._number_of_digits(max(len(block.segments) - 1, 0)))
        for i, segment in enumerate(block.segments):
            self.write_segment(segment=segment, name=name_pattern.format(i),
                               writer=writer, parent=segments_model)

    def write_segment(self, segment = None, name='0', writer=None, parent=None):
        """
        Write a Segment to the file

        :param segment: Segment to be written
        :param name: Name for segment representation in NSDF model tree (optional)
        :param writer: NSDFWriter instance (optional)
        :param parent: NSDF ModelComponent which will be the parent of segment NSDF representation (optional)
        """
        if not isinstance(segment, Segment):
            raise ValueError("Must provide a Segment to write.")

        if writer is None:
            writer = self._init_writing()

        single_segment = False
        if parent is None:
            neo_model, blocks_model, parent = self._prepare_model_tree()
            single_segment = True

        model = nsdf.ModelComponent(name, uid=uuid1().hex, parent=parent)

        self._write_segment_children(model, segment, writer)
        self._write_container_metadata(segment, model)

        if single_segment:
            writer.add_modeltree(neo_model)

    def _write_segment_children(self, model, segment, writer):
        analogsignals_model = nsdf.ModelComponent(name='analogsignals', uid=uuid1().hex, parent=model)
        name_pattern = '{{:0{}d}}'.format(self._number_of_digits(max(len(segment.analogsignals) - 1, 0)))
        for i, signal in enumerate(segment.analogsignals):
            self.write_analogsignal(signal=signal,
                                    name=name_pattern.format(i),
                                    parent=analogsignals_model, writer=writer)

    def write_analogsignal(self, signal, name='0', writer=None, parent=None):
        """
        Write an AnalogSignal to the file

        :param signal: AnalogSignal to be written
        :param name: Name for signal representation in NSDF model tree
        :param writer: NSDFWriter instance
        :param parent: NSDF ModelComponent which will be the parent of signal NSDF representation
        """
        uid = uuid1().hex
        model = nsdf.ModelComponent(name, uid=uid, parent=parent)

        r_signal = np.swapaxes(signal, 0, 1)
        channels, source_ds = self._create_signal_data_sources(model, r_signal, uid, writer)
        self._write_signal_data(model, channels, r_signal, signal, source_ds, writer)

        self._write_basic_metadata(model, signal)

    def _init_writing(self):
        return nsdf.NSDFWriter(self.filename, mode='w')

    def _prepare_model_tree(self):
        neo_model = nsdf.ModelComponent('neo', uid=uuid1().hex)
        blocks_model = nsdf.ModelComponent('blocks', uid=uuid1().hex, parent=neo_model)
        segments_model = nsdf.ModelComponent('segments', uid=uuid1().hex, parent=neo_model)
        return neo_model, blocks_model, segments_model

    def _number_of_digits(self, n):
        return len(str(n))

    def _name_pattern(self, how_many_items):
        return '{{:0{}d}}'.format(self._number_of_digits(max(how_many_items - 1, 0)))

    def _write_container_metadata(self, container, container_model):
        self._write_basic_metadata(container_model, container)

        self._write_datetime_attributes(container_model, container)
        self._write_index_attribute(container_model, container)

    def _write_basic_metadata(self, model, object):
        self._write_basic_attributes(model, object)
        self._write_annotations(model, object)

    def _write_basic_attributes(self, model, object):
        if object.name is not None:
            model.attrs['name'] = object.name
        if object.description is not None:
            model.attrs['description'] = object.description

    def _write_datetime_attributes(self, model, object):
        if object.rec_datetime is not None:
            model.attrs['rec_datetime'] = object.rec_datetime.strftime(self.dt_format)

    def _write_index_attribute(self, model, object):
        if object.index is not None:
            model.attrs['index'] = object.index

    def _write_annotations(self, model, object):
        if object.annotations is not None:
            model.attrs['annotations'] = pickle.dumps(object.annotations)

    def _write_signal_data(self, model, channels, r_signal, signal, source_ds, writer):
        dataobj = nsdf.UniformData('signal', unit=str(signal.units.dimensionality))
        dataobj.dtype = signal.dtype
        for i in range(len(channels)):
            dataobj.put_data(channels[i].uid, r_signal[i])

        dataobj.set_dt(float(signal.sampling_period.magnitude),
                       str(signal.sampling_period.dimensionality))

        self._write_analogsignal_t_start(dataobj, model, signal, source_ds, writer)

    def _write_analogsignal_t_start(self, dataobj, model, signal, source_ds, writer):
        rescaled_tstart = signal.t_start.rescale(signal.sampling_period.dimensionality)
        writer.add_uniform_data(source_ds, dataobj,
                                tstart=float(rescaled_tstart.magnitude))
        model.attrs['t_start_unit'] = str(signal.t_start.dimensionality)

    def _create_signal_data_sources(self, model, r_signal, uid, writer):
        channels = []
        channels_model = nsdf.ModelComponent(name='channels', uid=uuid1().hex, parent=model)
        name_pattern = '{{:0{}d}}'.format(self._number_of_digits(max(len(r_signal) - 1, 0)))
        for i, channel in enumerate(r_signal):
            channels.append(nsdf.ModelComponent(name_pattern.format(i),
                                                uid=uuid1().hex,
                                                parent=channels_model))

        source_ds = writer.add_uniform_ds(uid, [channel.uid for channel in channels])
        return channels, source_ds

    def read_all_blocks(self, lazy=False, cascade=True):
        """
        Read all blocks from the file

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :return: List of read blocks
        """
        reader = self._init_reading()
        blocks = []

        blocks_path = self.modeltree_path + 'blocks/'
        for block in reader.model[blocks_path].values():
            blocks.append(self.read_block(lazy, cascade, group=block, reader=reader))

        return blocks

    def read_block(self, lazy=False, cascade=True, group=None, reader=None):
        """
        Read a Block from the file

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param group: HDF5 Group representing the block in NSDF model tree (optional)
        :param reader: NSDFReader instance (optional)
        :return: Read block
        """
        block = Block()
        group, reader = self._select_first_container(group, reader, 'block')

        if group is None:
            return None

        attrs = group.attrs

        if cascade:
            self._read_block_children(lazy, block, group, reader)
        block.create_many_to_one_relationship()

        self._read_container_metadata(attrs, block)

        return block

    def _read_block_children(self, lazy, block, group, reader):
        for child in group['segments/'].values():
            block.segments.append(self.read_segment(lazy=lazy, group=child, reader=reader))

    def read_segment(self, lazy=False, cascade=True, group=None, reader=None):
        """
        Read a Segment from the file

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param group: HDF5 Group representing the block in NSDF model tree (optional)
        :param reader: NSDFReader instance (optional)
        :return: Read segment
        """
        segment = Segment()
        group, reader = self._select_first_container(group, reader, 'segment')

        if group is None:
            return None

        attrs = group.attrs

        if cascade:
            self._read_segment_children(lazy, group, reader, segment)

        self._read_container_metadata(attrs, segment)

        return segment

    def _read_segment_children(self, lazy, group, reader, segment):
        for child in group['analogsignals/'].values():
            segment.analogsignals.append(self.read_analogsignal(lazy=lazy, group=child, reader=reader))

    def read_analogsignal(self, lazy=False, cascade=True, group=None, reader=None):
        """
        Read an AnalogSignal from the file (must be child of a Segment)

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param group: HDF5 Group representing the block in NSDF model tree
        :param reader: NSDFReader instance
        :return: Read AnalogSignal
        """
        attrs = group.attrs
        uid = attrs['uid']
        data_group = reader.data['uniform/{}/signal'.format(uid)]

        t_start = self._read_analogsignal_t_start(attrs, data_group)
        signal = self._create_analogsignal(data_group, lazy, group, t_start, uid, reader)

        self._read_basic_metadata(attrs, signal)

        return signal

    def _init_reading(self):
        reader = nsdf.NSDFReader(self.filename)
        self.file_datetime = datetime.fromtimestamp(os.stat(self.filename).st_mtime)
        return reader

    def _select_first_container(self, group, reader, name):
        if reader is None:
            reader = self._init_reading()

        if group is None:
            path = self.modeltree_path + name + 's/'
            if len(reader.model[path].values()) > 0:
                group = reader.model[path].values()[0]

        return group, reader

    def _read_container_metadata(self, attrs, container):
        self._read_basic_metadata(attrs, container)

        self._read_datetime_attributes(attrs, container)
        self._read_index_attribute(attrs, container)

    def _read_basic_metadata(self, attrs, signal):
        self._read_basic_attributes(attrs, signal)
        self._read_annotations(attrs, signal)

    def _read_basic_attributes(self, attrs, object):
        if attrs.get('name') is not None:
            object.name = attrs['name']
        if attrs.get('description') is not None:
            object.description = attrs['description']
        object.file_origin = self.filename

    def _read_datetime_attributes(self, attrs, object):
        object.file_datetime = self.file_datetime
        if attrs.get('rec_datetime') is not None:
            object.rec_datetime = datetime.strptime(attrs['rec_datetime'], self.dt_format)

    def _read_annotations(self, attrs, object):
        if attrs.get('annotations') is not None:
            object.annotations = pickle.loads(attrs['annotations'])

    def _read_index_attribute(self, attrs, object):
        if attrs.get('index') is not None:
            object.index = attrs['index']

    def _create_analogsignal(self, data_group, lazy, group, t_start, uid, reader):
        if lazy:
            data_shape = data_group.shape
            data_shape = (data_shape[1], data_shape[0])
            signal = self._create_lazy_analogsignal(data_shape, data_group, uid, t_start)
        else:
            dataobj = reader.get_uniform_data(uid, 'signal')
            data = self._read_signal_data(dataobj, group)
            signal = self._create_normal_analogsignal(data, dataobj, uid, t_start)
        return signal

    def _read_analogsignal_t_start(self, attrs, data_group):
        t_start = float(data_group.attrs['tstart']) * pq.Quantity(1, data_group.attrs['tunit'])
        t_start = t_start.rescale(attrs['t_start_unit'])
        return t_start

    def _read_signal_data(self, dataobj, group):
        data = []
        for channel in group['channels/'].values():
            channel_uid = channel.attrs['uid']
            data += [dataobj.get_data(channel_uid)]
        return data

    def _create_normal_analogsignal(self, data, dataobj, uid, t_start):
        return AnalogSignal(np.swapaxes(data, 0, 1), dtype=dataobj.dtype, units=dataobj.unit,
                            t_start=t_start, sampling_period=pq.Quantity(dataobj.dt, dataobj.tunit))

    def _create_lazy_analogsignal(self, shape, data, uid, t_start):
        attrs = data.attrs
        signal = AnalogSignal([], dtype=data.dtype, units=attrs['unit'],
                              t_start=t_start, sampling_period=pq.Quantity(attrs['dt'], attrs['tunit']))
        signal.lazy_shape = shape
        return signal
