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


dt_format = '%d/%m/%Y %H:%M:%S'


class NSDFIO(BaseIO):
    """
    Class for reading and writing NSDF files
    """

    is_readable = True
    is_writable = True

    supported_objects = [Block, Segment, AnalogSignal]

    readable_objects = [Block]
    writeable_objects = [Block]

    has_header = False
    is_streameable = False

    name = 'nsdf'
    extensions = ['h5']
    mode = 'file'

    def __init__(self, filename = None):
        """
        Initialise NSDFIO instance

        :param filename: Path to the file
        """
        BaseIO.__init__(self)
        self.filename = filename

    def write_all_blocks(self, blocks):
        """
        Write list of blocks to the file

        :param blocks: List of blocks to be written
        """
        writer = nsdf.NSDFWriter(self.filename, mode='w')
        blocks_model, neo_model = self._prepare_model_tree()

        name_pattern = 'block_{{:0{}d}}'.format(self._number_of_digits(max(len(blocks) - 1, 0)))
        for i, block in enumerate(blocks):
            self.write_block(block, name_pattern.format(i), writer, blocks_model, neo_model)

        writer.add_modeltree(neo_model)

    def write_block(self, block, name = 'block_0', writer = None, blocks_model = None, neo_model = None):
        """
        Write single block to the file

        :param block: Block to be written
        :param writer: NSDFWriter instance (optional)
        :param blocks_model: NSDF model representing all blocks (optional)
        :param neo_model: NSDF model representing all data (optional)
        """
        if writer is None:
            writer = nsdf.NSDFWriter(self.filename, mode='w')

        single_block = False
        if blocks_model is None or neo_model is None:
            blocks_model, neo_model = self._prepare_model_tree()
            single_block = True

        uid = uuid1().hex
        block_model = nsdf.ModelComponent(name, uid = uid, parent = blocks_model)

        self._write_block_children(block, block_model, writer)

        self._write_basic_attributes(block_model, block)
        self._write_datetime_attributes(block_model, block)
        self._write_index_attribute(block_model, block)

        self._write_annotations(block_model, block)

        if single_block:
            writer.add_modeltree(neo_model)

    def _write_block_children(self, block, block_model, writer):
        name_pattern = 'segment_{{:0{}d}}'.format(self._number_of_digits(max(len(block.segments) - 1, 0)))
        for i, segment in enumerate(block.segments):
            self.write_segment(segment = segment,
                               name = name_pattern.format(i),
                               parent = block_model, writer = writer)

    def _number_of_digits(self, n):
        return len(str(n))

    def _prepare_model_tree(self):
        neo_model = nsdf.ModelComponent('neo', uid = uuid1().hex)
        blocks_model = nsdf.ModelComponent('blocks', uid = uuid1().hex, parent = neo_model)
        return blocks_model, neo_model

    def write_segment(self, segment, name, parent, writer):
        """
        Write a segment to the file (must be child of a block)

        :param segment: Segment to be written
        :param name: Name of the Segment in NSDF modeltree
        :param parent: Parent of the Segment in NSDF modeltree
        :param writer: NSDFWriter instance
        """
        uid = uuid1().hex
        model = nsdf.ModelComponent(name, uid = uid, parent = parent)

        self._write_segment_children(model, segment, writer)

        self._write_basic_attributes(model, segment)
        self._write_datetime_attributes(model, segment)
        self._write_index_attribute(model, segment)

        self._write_annotations(model, segment)

    def _write_segment_children(self, model, segment, writer):
        name_pattern = 'analogsignal_{{:0{}d}}'.format(self._number_of_digits(max(len(segment.analogsignals) - 1, 0)))
        for i, signal in enumerate(segment.analogsignals):
            self.write_analogsignal(signal = signal,
                                    name = name_pattern.format(i),
                                    parent = model, writer = writer)

    def _write_annotations(self, model, object):
        if object.annotations is not None:
            model.attrs['annotations'] = pickle.dumps(object.annotations)

    def _write_index_attribute(self, model, object):
        if object.index is not None:
            model.attrs['index'] = str(object.index)

    def _write_datetime_attributes(self, model, object):
        if object.file_datetime is not None:
            model.attrs['file_datetime'] = object.file_datetime.strftime(dt_format)
        if object.rec_datetime is not None:
            model.attrs['rec_datetime'] = object.rec_datetime.strftime(dt_format)

    def write_analogsignal(self, signal, name, parent, writer):
        """
        Write an AnalogSignal to the file (must be child of a segment)

        :param signal: AnalogSignal to be written
        :param name: Name of the AnalogSignal in NSDF modeltree
        :param parent: Parent of the AnalogSignal in NSDF modeltree
        :param writer: NSDFWriter instance
        """
        uid = uuid1().hex
        model = nsdf.ModelComponent(name, uid = uid, parent = parent)

        r_signal = np.swapaxes(signal, 0, 1)
        channels, source_ds = self._create_signal_data_sources(model, r_signal, uid, writer)
        self._write_signal_data(model, channels, r_signal, signal, source_ds, writer)

        self._write_basic_attributes(model, signal)
        self._write_annotations(model, signal)

    def _write_signal_data(self, model, channels, r_signal, signal, source_ds, writer):
        dataobj = nsdf.UniformData('signal', unit = str(signal.units.dimensionality))
        for i in range(len(channels)):
            dataobj.put_data(channels[i].uid, r_signal[i])
        dataobj.set_dt(float(signal.sampling_period.magnitude),
                       str(signal.sampling_period.dimensionality))
        writer.add_uniform_data(source_ds, dataobj,
                                tstart = float(signal.t_start.rescale(
                                    signal.sampling_period.dimensionality).magnitude))
        model.attrs['t_start_unit'] = str(signal.t_start.dimensionality)

    def _create_signal_data_sources(self, model, r_signal, uid, writer):
        channels = []
        name_pattern = 'channel_{{:0{}d}}'.format(self._number_of_digits(max(len(r_signal) - 1, 0)))
        for i, channel in enumerate(r_signal):
            channels.append(nsdf.ModelComponent(name_pattern.format(i),
                                                uid = uuid1().hex,
                                                parent = model))
        source_ds = writer.add_uniform_ds(uid, [channel.uid for channel in channels])
        return channels, source_ds

    def _write_basic_attributes(self, model, object):
        if object.name is not None:
            model.attrs['name'] = object.name
        if object.description is not None:
            model.attrs['description'] = object.description
        if object.file_origin is not None:
            model.attrs['file_origin'] = object.file_origin

    def read_all_blocks(self, lazy = False, cascade = True):
        """
        Read all blocks from the file

        :param lazy: Imposed by neo API (but not supported yet)
        :param cascade: Read nested objects or not?
        :return: list of read blocks
        """
        reader = nsdf.NSDFReader(self.filename)
        blocks = []

        for name in reader.model['/model/modeltree/neo/blocks'].keys():
            blocks.append(self.read_block(lazy, cascade, name = name, reader = reader))

        return blocks

    def read_block(self, lazy = False, cascade = True, name = None, reader = None):
        """
        Read a single block from the file

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param name: Name of block to be read (if not specified, first of all blocks is chosen)
        :param reader: NSDFReader instance (optional)
        :return: Read block
        """
        block = Block()
        name, reader = self._select_first_block(name, reader)

        if name is None:
            return None

        path = '/model/modeltree/neo/blocks/' + name
        attrs = reader.model[path].attrs

        if cascade:
            self._read_block_children(lazy, block, path, reader)
        block.create_many_to_one_relationship()

        self._read_basic_attributes(attrs, block)
        self._read_datetime_attributes(attrs, block)
        self._read_index_attribute(attrs, block)

        self._read_annotations(attrs, block)

        return block

    def _select_first_block(self, name, reader):
        if reader is None:
            reader = nsdf.NSDFReader(self.filename)
        if name is None and len(reader.model['/model/modeltree/neo/blocks'].keys()) != 0:
            name = reader.model['/model/modeltree/neo/blocks'].keys()[0]
        return name, reader

    def _read_datetime_attributes(self, attrs, object):
        if attrs.get('file_datetime') is not None:
            object.file_datetime = datetime.strptime(attrs['file_datetime'], dt_format)
        if attrs.get('rec_datetime') is not None:
            object.rec_datetime = datetime.strptime(attrs['rec_datetime'], dt_format)

    def _read_basic_attributes(self, attrs, object):
        if attrs.get('name') is not None:
            object.name = attrs['name']
        if attrs.get('description') is not None:
            object.description = attrs['description']
        if attrs.get('file_origin') is not None:
            object.file_origin = attrs['file_origin']

    def _read_block_children(self, lazy, block, path, reader):
        for nm in reader.model[path].keys():
            type = nm[:nm.find('_')]

            if type == 'segment':
                block.segments.append(self.read_segment(lazy = lazy,
                                                        path = path + '/' + nm,
                                                        reader = reader))

    def read_segment(self, lazy = False, cascade = True, path = None, reader = None):
        """
        Read a Segment from the file (must be child of a block)

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param path: Absolute path to Segment model in NSDF modeltree
        :param reader: NSDFReader instance
        :return: Read segment
        """
        attrs = reader.model[path].attrs
        uid = attrs['uid']
        segment = Segment()

        if cascade:
            self._read_segment_children(lazy, path, reader, segment)

        self._read_basic_attributes(attrs, segment)
        self._read_datetime_attributes(attrs, segment)
        self._read_index_attribute(attrs, segment)

        self._read_annotations(attrs, segment)

        return segment

    def _read_segment_children(self, lazy, path, reader, segment):
        for name in reader.model[path].keys():
            type = name[:name.find('_')]

            if type == 'analogsignal':
                segment.analogsignals.append(self.read_analogsignal(lazy = lazy,
                                                                    path = path + '/' + name,
                                                                    reader = reader))

    def _read_annotations(self, attrs, object):
        if attrs.get('annotations') is not None:
            object.annotations = pickle.loads(attrs['annotations'])

    def _read_index_attribute(self, attrs, object):
        if attrs.get('index') is not None:
            object.index = int(attrs['index'])

    def read_analogsignal(self, lazy = False, cascade = True, path = None, reader = None):
        """
        Read an AnalogSignal from the file (must be child of a Segment)

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param path: Absolute path to an AnalogSignal model in NSDF modeltree
        :param reader: NSDFReader instance
        :return: Read AnalogSignal
        """
        attrs = reader.model[path].attrs
        uid = attrs['uid']
        data_group = reader.data['uniform/{}/signal'.format(uid)]
        dataobj = reader.get_uniform_data(uid, 'signal')

        t_start = self._read_analogsignal_t_start(attrs, data_group, dataobj)
        signal = self._create_analogsignal(data_group, dataobj, lazy, path, reader, t_start, uid)

        self._read_basic_attributes(attrs, signal)
        self._read_annotations(attrs, signal)

        return signal

    def _create_analogsignal(self, data_group, dataobj, lazy, path, reader, t_start, uid):
        if lazy:
            data_shape = data_group.shape
            data_shape = (data_shape[1], data_shape[0])
            signal = self._create_lazy_analogsignal(data_shape, dataobj, reader, uid, t_start)
        else:
            data = self._read_signal_data(dataobj, path, reader)
            signal = self._create_normal_analogsignal(data, dataobj, reader, uid, t_start)
        return signal

    def _read_analogsignal_t_start(self, attrs, data_group, dataobj):
        t_start = float(data_group.attrs['tstart']) * pq.Quantity(1, dataobj.tunit)
        t_start = t_start.rescale(attrs['t_start_unit'])
        return t_start

    def _read_signal_data(self, dataobj, path, reader):
        data = []
        for name in reader.model[path].keys():
            channel_uid = reader.model[path + '/' + name].attrs['uid']
            data += [dataobj.get_data(channel_uid)]
        return data

    def _create_normal_analogsignal(self, data, dataobj, reader, uid, t_start):
        return AnalogSignal(np.swapaxes(data, 0, 1),
                            units = dataobj.unit,
                            t_start = t_start,
                            sampling_period = pq.Quantity(dataobj.dt, dataobj.tunit))

    def _create_lazy_analogsignal(self, shape, dataobj, reader, uid, t_start):
        signal =  AnalogSignal([],
                               units = dataobj.unit,
                               t_start = t_start,
                               sampling_period = pq.Quantity(dataobj.dt, dataobj.tunit))
        signal.lazy_shape = shape
        return signal