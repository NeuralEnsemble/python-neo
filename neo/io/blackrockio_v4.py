# -*- coding: utf-8 -*-
"""
Module for reading data from files in the Blackrock format.

This module is an older implementation  with old neo.io API.
A new class Blackrock compunded by BlackrockRawIO and BaseFromIO
superseed this one.

This work is based on:
  * Chris Rodgers - first version
  * Michael Denker, Lyuba Zehl - second version
  * Samuel Garcia - third version
  * Lyuba Zehl, Michael Denker - fourth version

This IO supports reading only.
This IO is able to read:
  * the nev file which contains spikes
  * ns1, ns2, .., ns6 files that contain signals at different sampling rates

This IO can handle the following Blackrock file specifications:
  * 2.1
  * 2.2
  * 2.3

The neural data channels are 1 - 128.
The analog inputs are 129 - 144. (129 - 137 AC coupled, 138 - 144 DC coupled)

spike- and event-data; 30000 Hz
"ns1": "analog data: 500 Hz",
"ns2": "analog data: 1000 Hz",
"ns3": "analog data: 2000 Hz",
"ns4": "analog data: 10000 Hz",
"ns5": "analog data: 30000 Hz",
"ns6": "analog data: 30000 Hz (no digital filter)"

TODO:
  * videosync events (file spec 2.3)
  * tracking events (file spec 2.3)
  * buttontrigger events (file spec 2.3)
  * config events (file spec 2.3)
  * check left sweep settings of Blackrock
  * check nsx offsets (file spec 2.1)
  * add info of nev ext header (NSASEXEX) to non-neural events
    (file spec 2.1 and 2.2)
  * read sif file information
  * read ccf file information
  * fix reading of periodic sampling events (non-neural event type)
    (file spec 2.1 and 2.2)
"""

from __future__ import division
import datetime
import os
import re

import numpy as np
import quantities as pq

import neo
from neo.io.baseio import BaseIO
from neo.core import (Block, Segment, SpikeTrain, Unit, Event,
                      ChannelIndex, AnalogSignal)

if __name__ == '__main__':
    pass


class BlackrockIO(BaseIO):
    """
    Class for reading data in from a file set recorded by the Blackrock
    (Cerebus) recording system.

    Upon initialization, the class is linked to the available set of Blackrock
    files. Data can be read as a neo Block or neo Segment object using the
    read_block or read_segment function, respectively.

    Note: This routine will handle files according to specification 2.1, 2.2,
    and 2.3. Recording pauses that may occur in file specifications 2.2 and
    2.3 are automatically extracted and the data set is split into different
    segments.

    Inherits from:
            neo.io.BaseIO

    The Blackrock data format consists not of a single file, but a set of
    different files. This constructor associates itself with a set of files
    that constitute a common data set. By default, all files belonging to
    the file set have the same base name, but different extensions.
    However, by using the override parameters, individual filenames can
    be set.

    Args:
        filename (string):
            File name (without extension) of the set of Blackrock files to
            associate with. Any .nsX or .nev, .sif, or .ccf extensions are
            ignored when parsing this parameter.
        nsx_override (string):
            File name of the .nsX files (without extension). If None,
            filename is used.
            Default: None.
        nev_override (string):
            File name of the .nev file (without extension). If None,
            filename is used.
            Default: None.
        sif_override (string):
            File name of the .sif file (without extension). If None,
            filename is used.
            Default: None.
        ccf_override (string):
            File name of the .ccf file (without extension). If None,
            filename is used.
            Default: None.
        verbose (boolean):
            If True, the class will output additional diagnostic
            information on stdout.
            Default: False

    Returns:
        -

    Examples:
        >>> a = BlackrockIO('myfile')

            Loads a set of file consisting of files myfile.ns1, ...,
            myfile.ns6, and myfile.nev


        >>> b = BlackrockIO('myfile', nev_override='sorted')

            Loads the analog data from the set of files myfile.ns1, ...,
            myfile.ns6, but reads spike/event data from sorted.nev
    """
    # Class variables demonstrating capabilities of this IO
    is_readable = True
    is_writable = False

    # This IO can only manipulate continuous data, spikes, and events
    supported_objects = [
        Block, Segment, Event, AnalogSignal, SpikeTrain, Unit, ChannelIndex]
    readable_objects = [Block, Segment]
    writeable_objects = []

    has_header = False
    is_streameable = False

    read_params = {
        neo.Block: [
            ('nsx_to_load', {
                'value': 'none',
                'label': "List of nsx files (ids, int) to read."}),
            ('n_starts', {
                'value': None,
                'label': "List of n_start points (Quantity) to create "
                         "segments from."}),
            ('n_stops', {
                'value': None,
                'label': "List of n_stop points (Quantity) to create "
                         "segments from."}),
            ('channels', {
                'value': 'none',
                'label': "List of channels (ids, int) to load data from."}),
            ('units', {
                'value': 'none',
                'label': "Dictionary for units (values, list of int) to load "
                         "for each channel (key, int)."}),
            ('load_waveforms', {
                'value': False,
                'label': "States if waveforms should be loaded and attached "
                         "to spiketrain"}),
            ('load_events', {
                'value': False,
                'label': "States if events should be loaded."})],
        neo.Segment: [
            ('n_start', {
                'label': "Start time point (Quantity) for segment"}),
            ('n_stop', {
                'label': "Stop time point (Quantity) for segment"}),
            ('nsx_to_load', {
                'value': 'none',
                'label': "List of nsx files (ids, int) to read."}),
            ('channels', {
                'value': 'none',
                'label': "List of channels (ids, int) to load data from."}),
            ('units', {
                'value': 'none',
                'label': "Dictionary for units (values, list of int) to load "
                         "for each channel (key, int)."}),
            ('load_waveforms', {
                'value': False,
                'label': "States if waveforms should be loaded and attached "
                         "to spiketrain"}),
            ('load_events', {
                'value': False,
                'label': "States if events should be loaded."})]}

    write_params = {}

    name = 'Blackrock IO'
    description = "This IO reads .nev/.nsX file of the Blackrock " + \
                  "(Cerebus) recordings system."
    # The possible file extensions of the Cerebus system and their content:
    #     ns1: contains analog data; sampled at 500 Hz (+ digital filters)
    #     ns2: contains analog data; sampled at 1000 Hz (+ digital filters)
    #     ns3: contains analog data; sampled at 2000 Hz (+ digital filters)
    #     ns4: contains analog data; sampled at 10000 Hz (+ digital filters)
    #     ns5: contains analog data; sampled at 30000 Hz (+ digital filters)
    #     ns6: contains analog data; sampled at 30000 Hz (no digital filters)
    #     nev: contains spike- and event-data; sampled at 30000 Hz
    #     sif: contains institution and patient info (XML)
    #     ccf: contains Cerebus configurations
    extensions = ['ns' + str(_) for _ in range(1, 7)]
    extensions.extend(['nev', 'sif', 'ccf'])

    mode = 'file'

    def __init__(self, filename, nsx_override=None, nev_override=None,
                 sif_override=None, ccf_override=None, verbose=False):
        """
        Initialize the BlackrockIO class.
        """
        BaseIO.__init__(self)

        # Used to avoid unnecessary repetition of verbose messages
        self.__verbose_messages = []

        # remove extension from base _filenames
        for ext in self.extensions:
            self.filename = re.sub(
                os.path.extsep + ext + '$', '', filename)

        # remove extensions from overrides
        self._filenames = {}
        if nsx_override:
            self._filenames['nsx'] = re.sub(
                os.path.extsep + 'ns[1,2,3,4,5,6]$', '', nsx_override)
        else:
            self._filenames['nsx'] = self.filename
        if nev_override:
            self._filenames['nev'] = re.sub(
                os.path.extsep + 'nev$', '', nev_override)
        else:
            self._filenames['nev'] = self.filename
        if sif_override:
            self._filenames['sif'] = re.sub(
                os.path.extsep + 'sif$', '', sif_override)
        else:
            self._filenames['sif'] = self.filename
        if ccf_override:
            self._filenames['ccf'] = re.sub(
                os.path.extsep + 'ccf$', '', ccf_override)
        else:
            self._filenames['ccf'] = self.filename

        # check which files are available
        self._avail_files = dict.fromkeys(self.extensions, False)

        self._avail_nsx = []
        for ext in self.extensions:
            if ext.startswith('ns'):
                file2check = ''.join(
                    [self._filenames['nsx'], os.path.extsep, ext])
            else:
                file2check = ''.join(
                    [self._filenames[ext], os.path.extsep, ext])

            if os.path.exists(file2check):
                self._print_verbose("Found " + file2check + ".")
                self._avail_files[ext] = True
                if ext.startswith('ns'):
                    self._avail_nsx.append(int(ext[-1]))

        # check if there are any files present
        if not any(list(self._avail_files.values())):
            raise IOError(
                'No Blackrock files present at {}'.format(filename))

        # check if manually specified files were found
        exts = ['nsx', 'nev', 'sif', 'ccf']
        ext_overrides = [nsx_override, nev_override, sif_override, ccf_override]
        for ext, ext_override in zip(exts, ext_overrides):
            if ext_override is not None and self._avail_files[ext] == False:
                raise ValueError('Specified {} file {} could not be '
                                 'found.'.format(ext, ext_override))

        # These dictionaries are used internally to map the file specification
        # revision of the nsx and nev files to one of the reading routines
        self.__nsx_header_reader = {
            '2.1': self.__read_nsx_header_variant_a,
            '2.2': self.__read_nsx_header_variant_b,
            '2.3': self.__read_nsx_header_variant_b}
        self.__nsx_dataheader_reader = {
            '2.1': self.__read_nsx_dataheader_variant_a,
            '2.2': self.__read_nsx_dataheader_variant_b,
            '2.3': self.__read_nsx_dataheader_variant_b}
        self.__nsx_data_reader = {
            '2.1': self.__read_nsx_data_variant_a,
            '2.2': self.__read_nsx_data_variant_b,
            '2.3': self.__read_nsx_data_variant_b}
        self.__nev_header_reader = {
            '2.1': self.__read_nev_header_variant_a,
            '2.2': self.__read_nev_header_variant_b,
            '2.3': self.__read_nev_header_variant_c}
        self.__nev_data_reader = {
            '2.1': self.__read_nev_data_variant_a,
            '2.2': self.__read_nev_data_variant_a,
            '2.3': self.__read_nev_data_variant_b}
        self.__nsx_params = {
            '2.1': self.__get_nsx_param_variant_a,
            '2.2': self.__get_nsx_param_variant_b,
            '2.3': self.__get_nsx_param_variant_b}
        self.__nsx_databl_param = {
            '2.1': self.__get_nsx_databl_param_variant_a,
            '2.2': self.__get_nsx_databl_param_variant_b,
            '2.3': self.__get_nsx_databl_param_variant_b}
        self.__waveform_size = {
            '2.1': self.__get_waveform_size_variant_a,
            '2.2': self.__get_waveform_size_variant_a,
            '2.3': self.__get_waveform_size_variant_b}
        self.__channel_labels = {
            '2.1': self.__get_channel_labels_variant_a,
            '2.2': self.__get_channel_labels_variant_b,
            '2.3': self.__get_channel_labels_variant_b}
        self.__nsx_rec_times = {
            '2.1': self.__get_nsx_rec_times_variant_a,
            '2.2': self.__get_nsx_rec_times_variant_b,
            '2.3': self.__get_nsx_rec_times_variant_b}
        self.__nonneural_evtypes = {
            '2.1': self.__get_nonneural_evtypes_variant_a,
            '2.2': self.__get_nonneural_evtypes_variant_a,
            '2.3': self.__get_nonneural_evtypes_variant_b}

        # Load file spec and headers of available nev file
        if self._avail_files['nev']:
            # read nev file specification
            self.__nev_spec = self.__extract_nev_file_spec()

            self._print_verbose('Specification Version ' + self.__nev_spec)

            # read nev headers
            self.__nev_basic_header, self.__nev_ext_header = \
                self.__nev_header_reader[self.__nev_spec]()

        # Load file spec and headers of available nsx files
        self.__nsx_spec = {}
        self.__nsx_basic_header = {}
        self.__nsx_ext_header = {}
        self.__nsx_data_header = {}
        for nsx_nb in self._avail_nsx:
            # read nsx file specification
            self.__nsx_spec[nsx_nb] = self.__extract_nsx_file_spec(nsx_nb)

            # read nsx headers
            self.__nsx_basic_header[nsx_nb], self.__nsx_ext_header[nsx_nb] = \
                self.__nsx_header_reader[self.__nsx_spec[nsx_nb]](nsx_nb)

            # Read nsx data header(s) for nsx
            self.__nsx_data_header[nsx_nb] = self.__nsx_dataheader_reader[
                self.__nsx_spec[nsx_nb]](nsx_nb)

    def _print_verbose(self, text):
        """
        Print a verbose diagnostic message (string).
        """
        if self.__verbose_messages:
            if text not in self.__verbose_messages:
                self.__verbose_messages.append(text)
                print(str(self.__class__.__name__) + ': ' + text)

    def __extract_nsx_file_spec(self, nsx_nb):
        """
        Extract file specification from an .nsx file.
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        # Header structure of files specification 2.2 and higher. For files 2.1
        # and lower, the entries ver_major and ver_minor are not supported.
        dt0 = [
            ('file_id', 'S8'),
            ('ver_major', 'uint8'),
            ('ver_minor', 'uint8')]

        nsx_file_id = np.fromfile(filename, count=1, dtype=dt0)[0]
        if nsx_file_id['file_id'].decode() == 'NEURALSG':
            spec = '2.1'
        elif nsx_file_id['file_id'].decode() == 'NEURALCD':
            spec = '{0}.{1}'.format(
                nsx_file_id['ver_major'], nsx_file_id['ver_minor'])
        else:
            raise IOError('Unsupported NSX file type.')

        return spec

    def __extract_nev_file_spec(self):
        """
        Extract file specification from an .nev file
        """
        filename = '.'.join([self._filenames['nev'], 'nev'])
        # Header structure of files specification 2.2 and higher. For files 2.1
        # and lower, the entries ver_major and ver_minor are not supported.
        dt0 = [
            ('file_id', 'S8'),
            ('ver_major', 'uint8'),
            ('ver_minor', 'uint8')]

        nev_file_id = np.fromfile(filename, count=1, dtype=dt0)[0]
        if nev_file_id['file_id'].decode() == 'NEURALEV':
            spec = '{0}.{1}'.format(
                nev_file_id['ver_major'], nev_file_id['ver_minor'])
        else:
            raise IOError('NEV file type {0} is not supported'.format(
                nev_file_id['file_id']))

        return spec

    def __read_nsx_header_variant_a(self, nsx_nb):
        """
        Extract nsx header information from a 2.1 .nsx file
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        # basic header (file_id: NEURALCD)
        dt0 = [
            ('file_id', 'S8'),
            # label of sampling groun (e.g. "1kS/s" or "LFP Low")
            ('label', 'S16'),
            # number of 1/30000 seconds between data points
            # (e.g., if sampling rate "1 kS/s", period equals "30")
            ('period', 'uint32'),
            ('channel_count', 'uint32')]

        nsx_basic_header = np.fromfile(filename, count=1, dtype=dt0)[0]

        # "extended" header (last field of file_id: NEURALCD)
        # (to facilitate compatibility with higher file specs)
        offset_dt0 = np.dtype(dt0).itemsize
        shape = nsx_basic_header['channel_count']
        # originally called channel_id in Blackrock user manual
        # (to facilitate compatibility with higher file specs)
        dt1 = [('electrode_id', 'uint32')]

        nsx_ext_header = np.memmap(
            filename, mode='r', shape=shape, offset=offset_dt0, dtype=dt1)

        return nsx_basic_header, nsx_ext_header

    def __read_nsx_header_variant_b(self, nsx_nb):
        """
        Extract nsx header information from a 2.2 or 2.3 .nsx file
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        # basic header (file_id: NEURALCD)
        dt0 = [
            ('file_id', 'S8'),
            # file specification split into major and minor version number
            ('ver_major', 'uint8'),
            ('ver_minor', 'uint8'),
            # bytes of basic & extended header
            ('bytes_in_headers', 'uint32'),
            # label of the sampling group (e.g., "1 kS/s" or "LFP low")
            ('label', 'S16'),
            ('comment', 'S256'),
            ('period', 'uint32'),
            ('timestamp_resolution', 'uint32'),
            # time origin: 2byte uint16 values for ...
            ('year', 'uint16'),
            ('month', 'uint16'),
            ('weekday', 'uint16'),
            ('day', 'uint16'),
            ('hour', 'uint16'),
            ('minute', 'uint16'),
            ('second', 'uint16'),
            ('millisecond', 'uint16'),
            # number of channel_count match number of extended headers
            ('channel_count', 'uint32')]

        nsx_basic_header = np.fromfile(filename, count=1, dtype=dt0)[0]

        # extended header (type: CC)
        offset_dt0 = np.dtype(dt0).itemsize
        shape = nsx_basic_header['channel_count']
        dt1 = [
            ('type', 'S2'),
            ('electrode_id', 'uint16'),
            ('electrode_label', 'S16'),
            # used front-end amplifier bank (e.g., A, B, C, D)
            ('physical_connector', 'uint8'),
            # used connector pin (e.g., 1-37 on bank A, B, C or D)
            ('connector_pin', 'uint8'),
            # digital and analog value ranges of the signal
            ('min_digital_val', 'int16'),
            ('max_digital_val', 'int16'),
            ('min_analog_val', 'int16'),
            ('max_analog_val', 'int16'),
            # units of the analog range values ("mV" or "uV")
            ('units', 'S16'),
            # filter settings used to create nsx from source signal
            ('hi_freq_corner', 'uint32'),
            ('hi_freq_order', 'uint32'),
            ('hi_freq_type', 'uint16'),  # 0=None, 1=Butterworth
            ('lo_freq_corner', 'uint32'),
            ('lo_freq_order', 'uint32'),
            ('lo_freq_type', 'uint16')]  # 0=None, 1=Butterworth

        nsx_ext_header = np.memmap(
            filename, mode='r', shape=shape, offset=offset_dt0, dtype=dt1)

        return nsx_basic_header, nsx_ext_header

    def __read_nsx_dataheader(self, nsx_nb, offset):
        """
        Reads data header following the given offset of an nsx file.
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        # dtypes data header
        dt2 = [
            ('header', 'uint8'),
            ('timestamp', 'uint32'),
            ('nb_data_points', 'uint32')]

        return np.memmap(
            filename, mode='r', dtype=dt2, shape=1, offset=offset)[0]

    def __read_nsx_dataheader_variant_a(
            self, nsx_nb, filesize=None, offset=None):
        """
        Reads None for the nsx data header of file spec 2.1. Introduced to
        facilitate compatibility with higher file spec.
        """

        return None

    def __read_nsx_dataheader_variant_b(
            self, nsx_nb, filesize=None, offset=None, ):
        """
        Reads the nsx data header for each data block following the offset of
        file spec 2.2 and 2.3.
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        filesize = self.__get_file_size(filename)

        data_header = {}
        index = 0

        if offset is None:
            offset = self.__nsx_basic_header[nsx_nb]['bytes_in_headers']

        while offset < filesize:
            index += 1

            dh = self.__read_nsx_dataheader(nsx_nb, offset)
            data_header[index] = {
                'header': dh['header'],
                'timestamp': dh['timestamp'],
                'nb_data_points': dh['nb_data_points'],
                'offset_to_data_block': offset + dh.dtype.itemsize}

            # data size = number of data points * (2bytes * number of channels)
            # use of `int` avoids overflow problem
            data_size = int(dh['nb_data_points']) * \
                        int(self.__nsx_basic_header[nsx_nb]['channel_count']) * 2
            # define new offset (to possible next data block)
            offset = data_header[index]['offset_to_data_block'] + data_size

        return data_header

    def __read_nsx_data_variant_a(self, nsx_nb):
        """
        Extract nsx data from a 2.1 .nsx file
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        # get shape of data
        shape = (
            self.__nsx_databl_param['2.1']('nb_data_points', nsx_nb),
            self.__nsx_basic_header[nsx_nb]['channel_count'])
        offset = self.__nsx_params['2.1']('bytes_in_headers', nsx_nb)

        # read nsx data
        # store as dict for compatibility with higher file specs
        data = {1: np.memmap(
            filename, mode='r', dtype='int16', shape=shape, offset=offset)}

        return data

    def __read_nsx_data_variant_b(self, nsx_nb):
        """
        Extract nsx data (blocks) from a 2.2 or 2.3 .nsx file. Blocks can arise
        if the recording was paused by the user.
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        data = {}
        for data_bl in self.__nsx_data_header[nsx_nb].keys():
            # get shape and offset of data
            shape = (
                self.__nsx_data_header[nsx_nb][data_bl]['nb_data_points'],
                self.__nsx_basic_header[nsx_nb]['channel_count'])
            offset = \
                self.__nsx_data_header[nsx_nb][data_bl]['offset_to_data_block']

            # read data
            data[data_bl] = np.memmap(
                filename, mode='r', dtype='int16', shape=shape, offset=offset)

        return data

    def __read_nev_header(self, ext_header_variants):
        """
        Extract nev header information from a 2.1 .nsx file
        """
        filename = '.'.join([self._filenames['nev'], 'nev'])

        # basic header
        dt0 = [
            # Set to "NEURALEV"
            ('file_type_id', 'S8'),
            ('ver_major', 'uint8'),
            ('ver_minor', 'uint8'),
            # Flags
            ('additionnal_flags', 'uint16'),
            # File index of first data sample
            ('bytes_in_headers', 'uint32'),
            # Number of bytes per data packet (sample)
            ('bytes_in_data_packets', 'uint32'),
            # Time resolution of time stamps in Hz
            ('timestamp_resolution', 'uint32'),
            # Sampling frequency of waveforms in Hz
            ('sample_resolution', 'uint32'),
            ('year', 'uint16'),
            ('month', 'uint16'),
            ('weekday', 'uint16'),
            ('day', 'uint16'),
            ('hour', 'uint16'),
            ('minute', 'uint16'),
            ('second', 'uint16'),
            ('millisecond', 'uint16'),
            ('application_to_create_file', 'S32'),
            ('comment_field', 'S256'),
            # Number of extended headers
            ('nb_ext_headers', 'uint32')]

        nev_basic_header = np.fromfile(filename, count=1, dtype=dt0)[0]

        # extended header
        # this consist in N block with code 8bytes + 24 data bytes
        # the data bytes depend on the code and need to be converted
        # cafilename_nsx, segse by case
        shape = nev_basic_header['nb_ext_headers']
        offset_dt0 = np.dtype(dt0).itemsize

        # This is the common structure of the beginning of extended headers
        dt1 = [
            ('packet_id', 'S8'),
            ('info_field', 'S24')]

        raw_ext_header = np.memmap(
            filename, mode='r', offset=offset_dt0, dtype=dt1, shape=shape)

        nev_ext_header = {}
        for packet_id in ext_header_variants.keys():
            mask = (raw_ext_header['packet_id'] == packet_id)
            dt2 = self.__nev_ext_header_types()[packet_id][
                ext_header_variants[packet_id]]

            nev_ext_header[packet_id] = raw_ext_header.view(dt2)[mask]
        return nev_basic_header, nev_ext_header

    def __read_nev_header_variant_a(self):
        """
        Extract nev header information from a 2.1 .nev file
        """

        ext_header_variants = {
            b'NEUEVWAV': 'a',
            b'ARRAYNME': 'a',
            b'ECOMMENT': 'a',
            b'CCOMMENT': 'a',
            b'MAPFILE': 'a',
            b'NSASEXEV': 'a'}

        return self.__read_nev_header(ext_header_variants)

    def __read_nev_header_variant_b(self):
        """
        Extract nev header information from a 2.2 .nev file
        """

        ext_header_variants = {
            b'NEUEVWAV': 'b',
            b'ARRAYNME': 'a',
            b'ECOMMENT': 'a',
            b'CCOMMENT': 'a',
            b'MAPFILE': 'a',
            b'NEUEVLBL': 'a',
            b'NEUEVFLT': 'a',
            b'DIGLABEL': 'a',
            b'NSASEXEV': 'a'}

        return self.__read_nev_header(ext_header_variants)

    def __read_nev_header_variant_c(self):
        """
        Extract nev header information from a 2.3 .nev file
        """

        ext_header_variants = {
            b'NEUEVWAV': 'b',
            b'ARRAYNME': 'a',
            b'ECOMMENT': 'a',
            b'CCOMMENT': 'a',
            b'MAPFILE': 'a',
            b'NEUEVLBL': 'a',
            b'NEUEVFLT': 'a',
            b'DIGLABEL': 'a',
            b'VIDEOSYN': 'a',
            b'TRACKOBJ': 'a'}

        return self.__read_nev_header(ext_header_variants)

    def __read_nev_data(self, nev_data_masks, nev_data_types):
        """
        Extract nev data from a 2.1 or 2.2 .nev file
        """
        filename = '.'.join([self._filenames['nev'], 'nev'])
        data_size = self.__nev_basic_header['bytes_in_data_packets']
        header_size = self.__nev_basic_header['bytes_in_headers']

        # read all raw data packets and markers
        dt0 = [
            ('timestamp', 'uint32'),
            ('packet_id', 'uint16'),
            ('value', 'S{0}'.format(data_size - 6))]

        raw_data = np.memmap(filename, mode='r', offset=header_size, dtype=dt0)

        masks = self.__nev_data_masks(raw_data['packet_id'])
        types = self.__nev_data_types(data_size)

        data = {}
        for k, v in nev_data_masks.items():
            data[k] = raw_data.view(types[k][nev_data_types[k]])[masks[k][v]]

        return data

    def __read_nev_data_variant_a(self):
        """
        Extract nev data from a 2.1 & 2.2 .nev file
        """
        nev_data_masks = {
            'NonNeural': 'a',
            'Spikes': 'a'}

        nev_data_types = {
            'NonNeural': 'a',
            'Spikes': 'a'}

        return self.__read_nev_data(nev_data_masks, nev_data_types)

    def __read_nev_data_variant_b(self):
        """
        Extract nev data from a 2.3 .nev file
        """
        nev_data_masks = {
            'NonNeural': 'a',
            'Spikes': 'b',
            'Comments': 'a',
            'VideoSync': 'a',
            'TrackingEvents': 'a',
            'ButtonTrigger': 'a',
            'ConfigEvent': 'a'}

        nev_data_types = {
            'NonNeural': 'b',
            'Spikes': 'a',
            'Comments': 'a',
            'VideoSync': 'a',
            'TrackingEvents': 'a',
            'ButtonTrigger': 'a',
            'ConfigEvent': 'a'}

        return self.__read_nev_data(nev_data_masks, nev_data_types)

    def __nev_ext_header_types(self):
        """
        Defines extended header types for different .nev file specifications.
        """
        nev_ext_header_types = {
            b'NEUEVWAV': {
                # Version>=2.1
                'a': [
                    ('packet_id', 'S8'),
                    ('electrode_id', 'uint16'),
                    ('physical_connector', 'uint8'),
                    ('connector_pin', 'uint8'),
                    ('digitization_factor', 'uint16'),
                    ('energy_threshold', 'uint16'),
                    ('hi_threshold', 'int16'),
                    ('lo_threshold', 'int16'),
                    ('nb_sorted_units', 'uint8'),
                    # number of bytes per waveform sample
                    ('bytes_per_waveform', 'uint8'),
                    ('unused', 'S10')],
                # Version>=2.3
                'b': [
                    ('packet_id', 'S8'),
                    ('electrode_id', 'uint16'),
                    ('physical_connector', 'uint8'),
                    ('connector_pin', 'uint8'),
                    ('digitization_factor', 'uint16'),
                    ('energy_threshold', 'uint16'),
                    ('hi_threshold', 'int16'),
                    ('lo_threshold', 'int16'),
                    ('nb_sorted_units', 'uint8'),
                    # number of bytes per waveform sample
                    ('bytes_per_waveform', 'uint8'),
                    # number of samples for each waveform
                    ('spike_width', 'uint16'),
                    ('unused', 'S8')]},
            b'ARRAYNME': {
                'a': [
                    ('packet_id', 'S8'),
                    ('electrode_array_name', 'S24')]},
            b'ECOMMENT': {
                'a': [
                    ('packet_id', 'S8'),
                    ('extra_comment', 'S24')]},
            b'CCOMMENT': {
                'a': [
                    ('packet_id', 'S8'),
                    ('continued_comment', 'S24')]},
            b'MAPFILE': {
                'a': [
                    ('packet_id', 'S8'),
                    ('mapFile', 'S24')]},
            b'NEUEVLBL': {
                'a': [
                    ('packet_id', 'S8'),
                    ('electrode_id', 'uint16'),
                    # label of this electrode
                    ('label', 'S16'),
                    ('unused', 'S6')]},
            b'NEUEVFLT': {
                'a': [
                    ('packet_id', 'S8'),
                    ('electrode_id', 'uint16'),
                    ('hi_freq_corner', 'uint32'),
                    ('hi_freq_order', 'uint32'),
                    # 0=None 1=Butterworth
                    ('hi_freq_type', 'uint16'),
                    ('lo_freq_corner', 'uint32'),
                    ('lo_freq_order', 'uint32'),
                    # 0=None 1=Butterworth
                    ('lo_freq_type', 'uint16'),
                    ('unused', 'S2')]},
            b'DIGLABEL': {
                'a': [
                    ('packet_id', 'S8'),
                    # Read name of digital
                    ('label', 'S16'),
                    # 0=serial, 1=parallel
                    ('mode', 'uint8'),
                    ('unused', 'S7')]},
            b'NSASEXEV': {
                'a': [
                    ('packet_id', 'S8'),
                    # Read frequency of periodic packet generation
                    ('frequency', 'uint16'),
                    # Read if digital input triggers events
                    ('digital_input_config', 'uint8'),
                    # Read if analog input triggers events
                    ('analog_channel_1_config', 'uint8'),
                    ('analog_channel_1_edge_detec_val', 'uint16'),
                    ('analog_channel_2_config', 'uint8'),
                    ('analog_channel_2_edge_detec_val', 'uint16'),
                    ('analog_channel_3_config', 'uint8'),
                    ('analog_channel_3_edge_detec_val', 'uint16'),
                    ('analog_channel_4_config', 'uint8'),
                    ('analog_channel_4_edge_detec_val', 'uint16'),
                    ('analog_channel_5_config', 'uint8'),
                    ('analog_channel_5_edge_detec_val', 'uint16'),
                    ('unused', 'S6')]},
            b'VIDEOSYN': {
                'a': [
                    ('packet_id', 'S8'),
                    ('video_source_id', 'uint16'),
                    ('video_source', 'S16'),
                    ('frame_rate', 'float32'),
                    ('unused', 'S2')]},
            b'TRACKOBJ': {
                'a': [
                    ('packet_id', 'S8'),
                    ('trackable_type', 'uint16'),
                    ('trackable_id', 'uint16'),
                    ('point_count', 'uint16'),
                    ('video_source', 'S16'),
                    ('unused', 'S2')]}}

        return nev_ext_header_types

    def __nev_data_masks(self, packet_ids):
        """
        Defines data masks for different .nev file specifications depending on
        the given packet identifiers.
        """
        __nev_data_masks = {
            'NonNeural': {
                'a': (packet_ids == 0)},
            'Spikes': {
                # Version 2.1 & 2.2
                'a': (0 < packet_ids) & (packet_ids <= 255),
                # Version>=2.3
                'b': (0 < packet_ids) & (packet_ids <= 2048)},
            'Comments': {
                'a': (packet_ids == 0xFFFF)},
            'VideoSync': {
                'a': (packet_ids == 0xFFFE)},
            'TrackingEvents': {
                'a': (packet_ids == 0xFFFD)},
            'ButtonTrigger': {
                'a': (packet_ids == 0xFFFC)},
            'ConfigEvent': {
                'a': (packet_ids == 0xFFFB)}}

        return __nev_data_masks

    def __nev_data_types(self, data_size):
        """
        Defines data types for different .nev file specifications depending on
        the given packet identifiers.
        """
        __nev_data_types = {
            'NonNeural': {
                # Version 2.1 & 2.2
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('packet_insertion_reason', 'uint8'),
                    ('reserved', 'uint8'),
                    ('digital_input', 'uint16'),
                    ('analog_input_channel_1', 'int16'),
                    ('analog_input_channel_2', 'int16'),
                    ('analog_input_channel_3', 'int16'),
                    ('analog_input_channel_4', 'int16'),
                    ('analog_input_channel_5', 'int16'),
                    ('unused', 'S{0}'.format(data_size - 20))],
                # Version>=2.3
                'b': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('packet_insertion_reason', 'uint8'),
                    ('reserved', 'uint8'),
                    ('digital_input', 'uint16'),
                    ('unused', 'S{0}'.format(data_size - 10))]},
            'Spikes': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('unit_class_nb', 'uint8'),
                    ('reserved', 'uint8'),
                    ('waveform', 'S{0}'.format(data_size - 8))]},
            'Comments': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('char_set', 'uint8'),
                    ('flag', 'uint8'),
                    ('data', 'uint32'),
                    ('comment', 'S{0}'.format(data_size - 12))]},
            'VideoSync': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('video_file_nb', 'uint16'),
                    ('video_frame_nb', 'uint32'),
                    ('video_elapsed_time', 'uint32'),
                    ('video_source_id', 'uint32'),
                    ('unused', 'int8', (data_size - 20,))]},
            'TrackingEvents': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('parent_id', 'uint16'),
                    ('node_id', 'uint16'),
                    ('node_count', 'uint16'),
                    ('point_count', 'uint16'),
                    ('tracking_points', 'uint16', ((data_size - 14) // 2,))]},
            'ButtonTrigger': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('trigger_type', 'uint16'),
                    ('unused', 'int8', (data_size - 8,))]},
            'ConfigEvent': {
                'a': [
                    ('timestamp', 'uint32'),
                    ('packet_id', 'uint16'),
                    ('config_change_type', 'uint16'),
                    ('config_changed', 'S{0}'.format(data_size - 8))]}}

        return __nev_data_types

    def __nev_params(self, param_name):
        """
        Returns wanted nev parameter.
        """
        nev_parameters = {
            'bytes_in_data_packets':
                self.__nev_basic_header['bytes_in_data_packets'],
            'rec_datetime': datetime.datetime(
                year=self.__nev_basic_header['year'],
                month=self.__nev_basic_header['month'],
                day=self.__nev_basic_header['day'],
                hour=self.__nev_basic_header['hour'],
                minute=self.__nev_basic_header['minute'],
                second=self.__nev_basic_header['second'],
                microsecond=self.__nev_basic_header['millisecond']),
            'max_res': self.__nev_basic_header['timestamp_resolution'],
            'channel_ids': self.__nev_ext_header[b'NEUEVWAV']['electrode_id'],
            'channel_labels': self.__channel_labels[self.__nev_spec](),
            'event_unit': pq.CompoundUnit("1.0/{0} * s".format(
                self.__nev_basic_header['timestamp_resolution'])),
            'nb_units': dict(zip(
                self.__nev_ext_header[b'NEUEVWAV']['electrode_id'],
                self.__nev_ext_header[b'NEUEVWAV']['nb_sorted_units'])),
            'digitization_factor': dict(zip(
                self.__nev_ext_header[b'NEUEVWAV']['electrode_id'],
                self.__nev_ext_header[b'NEUEVWAV']['digitization_factor'])),
            'data_size': self.__nev_basic_header['bytes_in_data_packets'],
            'waveform_size': self.__waveform_size[self.__nev_spec](),
            'waveform_dtypes': self.__get_waveforms_dtype(),
            'waveform_sampling_rate':
                self.__nev_basic_header['sample_resolution'] * pq.Hz,
            'waveform_time_unit': pq.CompoundUnit("1.0/{0} * s".format(
                self.__nev_basic_header['sample_resolution'])),
            'waveform_unit': pq.uV}

        return nev_parameters[param_name]

    def __get_file_size(self, filename):
        """
        Returns the file size in bytes for the given file.
        """
        filebuf = open(filename, 'rb')
        filebuf.seek(0, os.SEEK_END)
        file_size = filebuf.tell()
        filebuf.close()

        return file_size

    def __get_min_time(self):
        """
        Returns the smallest time that can be determined from the recording for
        use as the lower bound n in an interval [n,m).
        """
        tp = []
        if self._avail_files['nev']:
            tp.extend(self.__get_nev_rec_times()[0])
        for nsx_i in self._avail_nsx:
            tp.extend(self.__nsx_rec_times[self.__nsx_spec[nsx_i]](nsx_i)[0])

        return min(tp)

    def __get_max_time(self):
        """
        Returns the largest time that can be determined from the recording for
        use as the upper bound m in an interval [n,m).
        """
        tp = []
        if self._avail_files['nev']:
            tp.extend(self.__get_nev_rec_times()[1])
        for nsx_i in self._avail_nsx:
            tp.extend(self.__nsx_rec_times[self.__nsx_spec[nsx_i]](nsx_i)[1])

        return max(tp)

    def __get_nev_rec_times(self):
        """
        Extracts minimum and maximum time points from a nev file.
        """
        filename = '.'.join([self._filenames['nev'], 'nev'])

        dt = [('timestamp', 'uint32')]
        offset = \
            self.__get_file_size(filename) - \
            self.__nev_params('bytes_in_data_packets')
        last_data_packet = np.memmap(
            filename, mode='r', offset=offset, dtype=dt)[0]

        n_starts = [0 * self.__nev_params('event_unit')]
        n_stops = [
            last_data_packet['timestamp'] * self.__nev_params('event_unit')]

        return n_starts, n_stops

    def __get_nsx_rec_times_variant_a(self, nsx_nb):
        """
        Extracts minimum and maximum time points from a 2.1 nsx file.
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        t_unit = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'time_unit', nsx_nb)
        highest_res = self.__nev_params('event_unit')

        bytes_in_headers = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'bytes_in_headers', nsx_nb)
        nb_data_points = int(
            (self.__get_file_size(filename) - bytes_in_headers) /
            (2 * self.__nsx_basic_header[nsx_nb]['channel_count']) - 1)

        # add n_start
        n_starts = [(0 * t_unit).rescale(highest_res)]
        # add n_stop
        n_stops = [(nb_data_points * t_unit).rescale(highest_res)]

        return n_starts, n_stops

    def __get_nsx_rec_times_variant_b(self, nsx_nb):
        """
        Extracts minimum and maximum time points from a 2.2 or 2.3 nsx file.
        """
        t_unit = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'time_unit', nsx_nb)
        highest_res = self.__nev_params('event_unit')

        n_starts = []
        n_stops = []
        # add n-start and n_stop for all data blocks
        for data_bl in self.__nsx_data_header[nsx_nb].keys():
            ts0 = self.__nsx_data_header[nsx_nb][data_bl]['timestamp']
            nbdp = self.__nsx_data_header[nsx_nb][data_bl]['nb_data_points']

            # add n_start
            start = ts0 * t_unit
            n_starts.append(start.rescale(highest_res))
            # add n_stop
            stop = start + nbdp * t_unit
            n_stops.append(stop.rescale(highest_res))

        return sorted(n_starts), sorted(n_stops)

    def __get_waveforms_dtype(self):
        """
        Extracts the actual waveform dtype set for each channel.
        """
        # Blackrock code giving the approiate dtype
        conv = {0: 'int8', 1: 'int8', 2: 'int16', 4: 'int32'}

        # get all electrode ids from nev ext header
        all_el_ids = self.__nev_ext_header[b'NEUEVWAV']['electrode_id']

        # get the dtype of waveform (this is stupidly complicated)
        if self.__is_set(
                np.array(self.__nev_basic_header['additionnal_flags']), 0):
            dtype_waveforms = dict((k, 'int16') for k in all_el_ids)
        else:
            # extract bytes per waveform
            waveform_bytes = \
                self.__nev_ext_header[b'NEUEVWAV']['bytes_per_waveform']
            # extract dtype for waveforms fro each electrode
            dtype_waveforms = dict(zip(all_el_ids, conv[waveform_bytes]))

        return dtype_waveforms

    def __get_channel_labels_variant_a(self):
        """
        Returns labels for all channels for file spec 2.1
        """
        elids = self.__nev_ext_header[b'NEUEVWAV']['electrode_id']
        labels = []

        for elid in elids:
            if elid < 129:
                labels.append('chan%i' % elid)
            else:
                labels.append('ainp%i' % (elid - 129 + 1))

        return dict(zip(elids, labels))

    def __get_channel_labels_variant_b(self):
        """
        Returns labels for all channels for file spec 2.2 and 2.3
        """
        elids = self.__nev_ext_header[b'NEUEVWAV']['electrode_id']
        labels = self.__nev_ext_header[b'NEUEVLBL']['label']

        return dict(zip(elids, labels)) if len(labels) > 0 else None

    def __get_waveform_size_variant_a(self):
        """
        Returns wavform sizes for all channels for file spec 2.1 and 2.2
        """
        wf_dtypes = self.__get_waveforms_dtype()
        nb_bytes_wf = self.__nev_basic_header['bytes_in_data_packets'] - 8

        wf_sizes = dict([
            (ch, int(nb_bytes_wf / np.dtype(dt).itemsize)) for ch, dt in
            wf_dtypes.items()])

        return wf_sizes

    def __get_waveform_size_variant_b(self):
        """
        Returns wavform sizes for all channels for file spec 2.3
        """
        elids = self.__nev_ext_header[b'NEUEVWAV']['electrode_id']
        spike_widths = self.__nev_ext_header[b'NEUEVWAV']['spike_width']

        return dict(zip(elids, spike_widths))

    def __get_left_sweep_waveforms(self):
        """
        Returns left sweep of waveforms for each channel. Left sweep is defined
        as the time from the beginning of the waveform to the trigger time of
        the corresponding spike.
        """
        # TODO: Double check if this is the actual setting for Blackrock
        wf_t_unit = self.__nev_params('waveform_time_unit')
        all_ch = self.__nev_params('channel_ids')

        # TODO: Double check if this is the correct assumption (10 samples)
        # default value: threshold crossing after 10 samples of waveform
        wf_left_sweep = dict([(ch, 10 * wf_t_unit) for ch in all_ch])

        # non-default: threshold crossing at center of waveform
        # wf_size = self.__nev_params('waveform_size')
        # wf_left_sweep = dict(
        #     [(ch, (wf_size[ch] / 2) * wf_t_unit) for ch in all_ch])

        return wf_left_sweep

    def __get_nsx_param_variant_a(self, param_name, nsx_nb):
        """
        Returns parameter (param_name) for a given nsx (nsx_nb) for file spec
        2.1.
        """
        # Here, min/max_analog_val and min/max_digital_val are not available in
        # the nsx, so that we must estimate these parameters from the
        # digitization factor of the nev (information by Kian Torab, Blackrock
        # Microsystems). Here dig_factor=max_analog_val/max_digital_val. We set
        # max_digital_val to 1000, and max_analog_val=dig_factor. dig_factor is
        # given in nV by definition, so the units turn out to be uV.
        labels = []
        dig_factor = []
        for elid in self.__nsx_ext_header[nsx_nb]['electrode_id']:
            if self._avail_files['nev']:
                # This is a workaround for the DigitalFactor overflow in NEV
                # files recorded with buggy Cerebus system.
                # Fix taken from: NMPK toolbox by Blackrock,
                # file openNEV, line 464,
                # git rev. d0a25eac902704a3a29fa5dfd3aed0744f4733ed
                df = self.__nev_params('digitization_factor')[elid]
                if df == 21516:
                    df = 152592.547
                dig_factor.append(df)
            else:
                dig_factor.append(None)

            if elid < 129:
                labels.append('chan%i' % elid)
            else:
                labels.append('ainp%i' % (elid - 129 + 1))

        nsx_parameters = {
            'labels': labels,
            'units': np.array(
                ['uV'] *
                self.__nsx_basic_header[nsx_nb]['channel_count']),
            'min_analog_val': -1 * np.array(dig_factor),
            'max_analog_val': np.array(dig_factor),
            'min_digital_val': np.array(
                [-1000] * self.__nsx_basic_header[nsx_nb]['channel_count']),
            'max_digital_val': np.array(
                [1000] * self.__nsx_basic_header[nsx_nb]['channel_count']),
            'timestamp_resolution': 30000,
            'bytes_in_headers':
                self.__nsx_basic_header[nsx_nb].dtype.itemsize +
                self.__nsx_ext_header[nsx_nb].dtype.itemsize *
                self.__nsx_basic_header[nsx_nb]['channel_count'],
            'sampling_rate':
                30000 / self.__nsx_basic_header[nsx_nb]['period'] * pq.Hz,
            'time_unit': pq.CompoundUnit("1.0/{0}*s".format(
                30000 / self.__nsx_basic_header[nsx_nb]['period']))}

        return nsx_parameters[param_name]

    def __get_nsx_param_variant_b(self, param_name, nsx_nb):
        """
        Returns parameter (param_name) for a given nsx (nsx_nb) for file spec
        2.2 and 2.3.
        """
        nsx_parameters = {
            'labels':
                self.__nsx_ext_header[nsx_nb]['electrode_label'],
            'units':
                self.__nsx_ext_header[nsx_nb]['units'],
            'min_analog_val':
                self.__nsx_ext_header[nsx_nb]['min_analog_val'],
            'max_analog_val':
                self.__nsx_ext_header[nsx_nb]['max_analog_val'],
            'min_digital_val':
                self.__nsx_ext_header[nsx_nb]['min_digital_val'],
            'max_digital_val':
                self.__nsx_ext_header[nsx_nb]['max_digital_val'],
            'timestamp_resolution':
                self.__nsx_basic_header[nsx_nb]['timestamp_resolution'],
            'bytes_in_headers':
                self.__nsx_basic_header[nsx_nb]['bytes_in_headers'],
            'sampling_rate':
                self.__nsx_basic_header[nsx_nb]['timestamp_resolution'] /
                self.__nsx_basic_header[nsx_nb]['period'] * pq.Hz,
            'time_unit': pq.CompoundUnit("1.0/{0}*s".format(
                self.__nsx_basic_header[nsx_nb]['timestamp_resolution'] /
                self.__nsx_basic_header[nsx_nb]['period']))}

        return nsx_parameters[param_name]

    def __get_nsx_databl_param_variant_a(
            self, param_name, nsx_nb, n_start=None, n_stop=None):
        """
        Returns data block parameter (param_name) for a given nsx (nsx_nb) for
        file spec 2.1. Arg 'n_start' should not be specified! It is only set
        for compatibility reasons with higher file spec.
        """
        filename = '.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb])

        t_starts, t_stops = \
            self.__nsx_rec_times[self.__nsx_spec[nsx_nb]](nsx_nb)

        bytes_in_headers = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'bytes_in_headers', nsx_nb)

        # extract parameters from nsx basic extended and data header
        data_parameters = {
            'nb_data_points': int(
                (self.__get_file_size(filename) - bytes_in_headers) /
                (2 * self.__nsx_basic_header[nsx_nb]['channel_count']) - 1),
            'databl_idx': 1,
            'databl_t_start': t_starts[0],
            'databl_t_stop': t_stops[0]}

        return data_parameters[param_name]

    def __get_nsx_databl_param_variant_b(
            self, param_name, nsx_nb, n_start, n_stop):
        """
        Returns data block parameter (param_name) for a given nsx (nsx_nb) with
        a wanted n_start for file spec 2.2 and 2.3.
        """
        t_starts, t_stops = \
            self.__nsx_rec_times[self.__nsx_spec[nsx_nb]](nsx_nb)

        # data header
        for d_bl in self.__nsx_data_header[nsx_nb].keys():
            # from "data header" with corresponding t_start and t_stop
            data_parameters = {
                'nb_data_points':
                    self.__nsx_data_header[nsx_nb][d_bl]['nb_data_points'],
                'databl_idx': d_bl,
                'databl_t_start': t_starts[d_bl - 1],
                'databl_t_stop': t_stops[d_bl - 1]}
            if t_starts[d_bl - 1] <= n_start < n_stop <= t_stops[d_bl - 1]:
                return data_parameters[param_name]
            elif n_start < t_starts[d_bl - 1] < n_stop <= t_stops[d_bl - 1]:
                self._print_verbose(
                    "User n_start ({0}) is smaller than the corresponding "
                    "t_start of the available ns{1} datablock "
                    "({2}).".format(n_start, nsx_nb, t_starts[d_bl - 1]))
                return data_parameters[param_name]
            elif t_starts[d_bl - 1] <= n_start < t_stops[d_bl - 1] < n_stop:
                self._print_verbose(
                    "User n_stop ({0}) is larger than the corresponding "
                    "t_stop of the available ns{1} datablock "
                    "({2}).".format(n_stop, nsx_nb, t_stops[d_bl - 1]))
                return data_parameters[param_name]
            elif n_start < t_starts[d_bl - 1] < t_stops[d_bl - 1] < n_stop:
                self._print_verbose(
                    "User n_start ({0}) is smaller than the corresponding "
                    "t_start and user n_stop ({1}) is larger than the "
                    "corresponding t_stop of the available ns{2} datablock "
                    "({3}).".format(
                        n_start, n_stop, nsx_nb,
                        (t_starts[d_bl - 1], t_stops[d_bl - 1])))
                return data_parameters[param_name]
            else:
                continue

        raise ValueError(
            "User n_start and n_stop are all smaller or larger than the "
            "t_start and t_stops of all available ns%i datablocks" % nsx_nb)

    def __get_nonneural_evtypes_variant_a(self, data):
        """
        Defines event types and the necessary parameters to extract them from
        a 2.1 and 2.2 nev file.
        """
        # TODO: add annotations of nev ext header (NSASEXEX) to event types

        # digital events
        event_types = {
            'digital_input_port': {
                'name': 'digital_input_port',
                'field': 'digital_input',
                'mask': self.__is_set(data['packet_insertion_reason'], 0),
                'desc': "Events of the digital input port"},
            'serial_input_port': {
                'name': 'serial_input_port',
                'field': 'digital_input',
                'mask':
                    self.__is_set(data['packet_insertion_reason'], 0) &
                    self.__is_set(data['packet_insertion_reason'], 7),
                'desc': "Events of the serial input port"}}

        # analog input events via threshold crossings
        for ch in range(5):
            event_types.update({
                'analog_input_channel_{0}'.format(ch + 1): {
                    'name': 'analog_input_channel_{0}'.format(ch + 1),
                    'field': 'analog_input_channel_{0}'.format(ch + 1),
                    'mask': self.__is_set(
                        data['packet_insertion_reason'], ch + 1),
                    'desc': "Values of analog input channel {0} in mV "
                            "(+/- 5000)".format(ch + 1)}})

        # TODO: define field and desc
        event_types.update({
            'periodic_sampling_events': {
                'name': 'periodic_sampling_events',
                'field': 'digital_input',
                'mask': self.__is_set(data['packet_insertion_reason'], 6),
                'desc': 'Periodic sampling event of a certain frequency'}})

        return event_types

    def __get_nonneural_evtypes_variant_b(self, data):
        """
        Defines event types and the necessary parameters to extract them from
        a 2.3 nev file.
        """
        # digital events
        event_types = {
            'digital_input_port': {
                'name': 'digital_input_port',
                'field': 'digital_input',
                'mask': self.__is_set(data['packet_insertion_reason'], 0),
                'desc': "Events of the digital input port"},
            'serial_input_port': {
                'name': 'serial_input_port',
                'field': 'digital_input',
                'mask':
                    self.__is_set(data['packet_insertion_reason'], 0) &
                    self.__is_set(data['packet_insertion_reason'], 7),
                'desc': "Events of the serial input port"}}

        return event_types

    def __get_unit_classification(self, un_id):
        """
        Returns the Blackrock unit classification of an online spike sorting
        for the given unit id (un_id).
        """
        # Blackrock unit classification
        if un_id == 0:
            return 'unclassified'
        elif 1 <= un_id <= 16:
            return '{0}'.format(un_id)
        elif 17 <= un_id <= 244:
            raise ValueError(
                "Unit id {0} is not used by daq system".format(un_id))
        elif un_id == 255:
            return 'noise'
        else:
            raise ValueError("Unit id {0} cannot be classified".format(un_id))

    def __is_set(self, flag, pos):
        """
        Checks if bit is set at the given position for flag. If flag is an
        array, an array will be returned.
        """
        return flag & (1 << pos) > 0

    def __transform_nsx_to_load(self, nsx_to_load):
        """
        Transforms the input argument nsx_to_load to a list of integers.
        """
        if hasattr(nsx_to_load, "__len__") and len(nsx_to_load) == 0:
            nsx_to_load = None
        if isinstance(nsx_to_load, int):
            nsx_to_load = [nsx_to_load]
        if isinstance(nsx_to_load, str):
            if nsx_to_load.lower() == 'none':
                nsx_to_load = None
            elif nsx_to_load.lower() == 'all':
                nsx_to_load = self._avail_nsx
            else:
                raise ValueError("Invalid specification of nsx_to_load.")

        if nsx_to_load:
            for nsx_nb in nsx_to_load:
                if not self._avail_files['ns' + str(nsx_nb)]:
                    raise ValueError("ns%i is not available" % nsx_nb)

        return nsx_to_load

    def __transform_channels(self, channels, nsx_to_load):
        """
        Transforms the input argument channels to a list of integers.
        """
        all_channels = []
        nsx_to_load = self.__transform_nsx_to_load(nsx_to_load)
        if nsx_to_load is not None:
            for nsx_nb in nsx_to_load:
                all_channels.extend(
                    self.__nsx_ext_header[nsx_nb]['electrode_id'].astype(int))
        elec_id = self.__nev_ext_header[b'NEUEVWAV']['electrode_id']
        all_channels.extend(elec_id.astype(int))
        all_channels = np.unique(all_channels).tolist()

        if hasattr(channels, "__len__") and len(channels) == 0:
            channels = None
        if isinstance(channels, int):
            channels = [channels]
        if isinstance(channels, str):
            if channels.lower() == 'none':
                channels = None
            elif channels.lower() == 'all':
                channels = all_channels
            else:
                raise ValueError("Invalid channel specification.")

        if channels:
            if len(set(all_channels) & set(channels)) < len(channels):
                self._print_verbose(
                    "Ignoring unknown channel ID(s) specified in in channels.")

            # Make sure, all channels are valid and contain no duplicates
            channels = list(set(all_channels).intersection(set(channels)))
        else:
            self._print_verbose("No channel is specified, therefore no "
                                "time series and unit data is loaded.")

        return channels

    def __transform_units(self, units, channels):
        """
        Transforms the input argument nsx_to_load to a dictionary, where keys
        (channels) are int, and values (units) are lists of integers.
        """
        if isinstance(units, dict):
            for ch, u in units.items():
                if ch not in channels:
                    self._print_verbose(
                        "Units contain a channel id which is not listed in "
                        "channels")
                if isinstance(u, int):
                    units[ch] = [u]
                if hasattr(u, '__len__') and len(u) == 0:
                    units[ch] = None
                if isinstance(u, str):
                    if u.lower() == 'none':
                        units[ch] = None
                    elif u.lower() == 'all':
                        units[ch] = list(range(17))
                        units[ch].append(255)
                    else:
                        raise ValueError("Invalid unit specification.")
        else:
            if hasattr(units, "__len__") and len(units) == 0:
                units = None
            if isinstance(units, str):
                if units.lower() == 'none':
                    units = None
                elif units.lower() == 'all':
                    units = list(range(17))
                    units.append(255)
                else:
                    raise ValueError("Invalid unit specification.")
            if isinstance(units, int):
                units = [units]

            if (channels is None) and (units is not None):
                raise ValueError(
                    'At least one channel needs to be loaded to load units')

            if units:
                units = dict(zip(channels, [units] * len(channels)))

        if units is None:
            self._print_verbose("No units are specified, therefore no "
                                "unit or spiketrain is loaded.")

        return units

    def __transform_times(self, n, default_n):
        """
        Transforms the input argument n_start or n_stop (n) to a list of
        quantities. In case n is None, it is set to a default value provided by
        the given function (default_n).
        """
        highest_res = self.__nev_params('event_unit')

        if isinstance(n, pq.Quantity):
            n = [n.rescale(highest_res)]
        elif hasattr(n, "__len__"):
            n = [tp.rescale(highest_res) if tp is not None
                 else default_n for tp in n]
        elif n is None:
            n = [default_n]
        else:
            raise ValueError('Invalid specification of n_start/n_stop.')

        return n

    def __merge_time_ranges(
            self, user_n_starts, user_n_stops, nsx_to_load):
        """
        Merges after a validation the user specified n_starts and n_stops with
        the intrinsically given n_starts and n_stops (from e.g, recording
        pauses) of the file set.

        Final n_starts and n_stops are chosen, so that the time range of each
        resulting segment is set to the best meaningful maximum. This means
        that the duration of the signals stored in the segments might be
        smaller than the actually set duration of the segment.
        """

        # define the higest time resolution
        # (for accurate manipulations of the time settings)
        max_time = self.__get_max_time()
        min_time = self.__get_min_time()
        highest_res = self.__nev_params('event_unit')
        user_n_starts = self.__transform_times(
            user_n_starts, min_time)
        user_n_stops = self.__transform_times(
            user_n_stops, max_time)

        # check if user provided as many n_starts as n_stops
        if len(user_n_starts) != len(user_n_stops):
            raise ValueError("n_starts and n_stops must be of equal length")

        # if necessary reset max n_stop to max time of file set
        start_stop_id = 0
        while start_stop_id < len(user_n_starts):
            if user_n_starts[start_stop_id] < min_time:
                user_n_starts[start_stop_id] = min_time
                self._print_verbose(
                    "Entry of n_start '{}' is smaller than min time of the file "
                    "set: n_start set to min time of file set"
                    "".format(user_n_starts[start_stop_id]))
            if user_n_stops[start_stop_id] > max_time:
                user_n_stops[start_stop_id] = max_time
                self._print_verbose(
                    "Entry of n_stop '{}' is larger than max time of the file "
                    "set: n_stop set to max time of file set"
                    "".format(user_n_stops[start_stop_id]))

            if (user_n_stops[start_stop_id] < min_time
                or user_n_starts[start_stop_id] > max_time):
                user_n_stops.pop(start_stop_id)
                user_n_starts.pop(start_stop_id)
                self._print_verbose(
                    "Entry of n_start is larger than max time or entry of "
                    "n_stop is smaller than min time of the "
                    "file set: n_start and n_stop are ignored")
                continue
            start_stop_id += 1

        # get intrinsic time settings of nsx files (incl. rec pauses)
        n_starts_files = []
        n_stops_files = []
        if nsx_to_load is not None:
            for nsx_nb in nsx_to_load:
                start_stop = \
                    self.__nsx_rec_times[self.__nsx_spec[nsx_nb]](nsx_nb)
                n_starts_files.append(start_stop[0])
                n_stops_files.append(start_stop[1])

        # reducing n_starts from wanted nsx files to minima
        # (keep recording pause if it occurs)
        if len(n_starts_files) > 0:
            if np.shape(n_starts_files)[1] > 1:
                n_starts_files = [
                    tp * highest_res for tp in np.min(n_starts_files, axis=1)]
            else:
                n_starts_files = [
                    tp * highest_res for tp in np.min(n_starts_files, axis=0)]

        # reducing n_starts from wanted nsx files to maxima
        # (keep recording pause if it occurs)
        if len(n_stops_files) > 0:
            if np.shape(n_stops_files)[1] > 1:
                n_stops_files = [
                    tp * highest_res for tp in np.max(n_stops_files, axis=1)]
            else:
                n_stops_files = [
                    tp * highest_res for tp in np.max(n_stops_files, axis=0)]

        # merge user time settings with intrinsic nsx time settings
        n_starts = []
        n_stops = []
        for start, stop in zip(user_n_starts, user_n_stops):
            # check if start and stop of user create a positive time interval
            if not start < stop:
                raise ValueError(
                    "t(i) in n_starts has to be smaller than t(i) in n_stops")

            # Reduce n_starts_files to given intervals of user & add start
            if len(n_starts_files) > 0:
                mask = (n_starts_files > start) & (n_starts_files < stop)
                red_n_starts_files = np.array(n_starts_files)[mask]
                merged_n_starts = [start] + [
                    tp * highest_res for tp in red_n_starts_files]
            else:
                merged_n_starts = [start]

            # Reduce n_stops_files to given intervals of user & add stop
            if len(n_stops_files) > 0:
                mask = (n_stops_files > start) & (n_stops_files < stop)
                red_n_stops_files = np.array(n_stops_files)[mask]
                merged_n_stops = [
                                     tp * highest_res for tp in red_n_stops_files] + [stop]
            else:
                merged_n_stops = [stop]
            # Define combined user and file n_starts and n_stops
            # case one:
            if len(merged_n_starts) == len(merged_n_stops):
                if len(merged_n_starts) + len(merged_n_stops) == 2:
                    n_starts.extend(merged_n_starts)
                    n_stops.extend(merged_n_stops)
                if len(merged_n_starts) + len(merged_n_stops) > 2:
                    merged_n_starts.remove(merged_n_starts[1])
                    n_starts.extend([merged_n_starts])
                    merged_n_stops.remove(merged_n_stops[-2])
                    n_stops.extend(merged_n_stops)
            # case two:
            elif len(merged_n_starts) < len(merged_n_stops):
                n_starts.extend(merged_n_starts)
                merged_n_stops.remove(merged_n_stops[-2])
                n_stops.extend(merged_n_stops)
            # case three:
            elif len(merged_n_starts) > len(merged_n_stops):
                merged_n_starts.remove(merged_n_starts[1])
                n_starts.extend(merged_n_starts)
                n_stops.extend(merged_n_stops)

        if len(n_starts) > len(user_n_starts) and \
                        len(n_stops) > len(user_n_stops):
            self._print_verbose(
                "Additional recording pauses were detected. There will be "
                "more segments than the user expects.")

        return n_starts, n_stops

    def __read_event(self, n_start, n_stop, data, ev_dict, lazy=False):
        """
        Creates an event for non-neural experimental events in nev data.
        """
        event_unit = self.__nev_params('event_unit')

        if lazy:
            times = []
            labels = np.array([], dtype='S')
        else:
            times = data['timestamp'][ev_dict['mask']] * event_unit
            labels = data[ev_dict['field']][ev_dict['mask']].astype(str)

        # mask for given time interval
        mask = (times >= n_start) & (times < n_stop)
        if np.sum(mask) > 0:
            ev = Event(
                times=times[mask].astype(float),
                labels=labels[mask],
                name=ev_dict['name'],
                description=ev_dict['desc'])
            if lazy:
                ev.lazy_shape = np.sum(mask)
        else:
            ev = None

        return ev

    def __read_spiketrain(
            self, n_start, n_stop, spikes, channel_id, unit_id,
            load_waveforms=False, scaling='raw', lazy=False):
        """
        Creates spiketrains for Spikes in nev data.
        """
        event_unit = self.__nev_params('event_unit')

        # define a name for spiketrain
        # (unique identifier: 1000 * elid + unit_nb)
        name = "Unit {0}".format(1000 * channel_id + unit_id)
        # define description for spiketrain
        desc = 'SpikeTrain from channel: {0}, unit: {1}'.format(
            channel_id, self.__get_unit_classification(unit_id))

        # get spike times for given time interval
        if not lazy:
            times = spikes['timestamp'] * event_unit
            mask = (times >= n_start) & (times <= n_stop)
            times = times[mask].astype(float)
        else:
            times = np.array([]) * event_unit

        st = SpikeTrain(
            times=times,
            name=name,
            description=desc,
            file_origin='.'.join([self._filenames['nev'], 'nev']),
            t_start=n_start,
            t_stop=n_stop)

        if lazy:
            st.lazy_shape = np.shape(times)

        # load waveforms if requested
        if load_waveforms and not lazy:
            wf_dtype = self.__nev_params('waveform_dtypes')[channel_id]
            wf_size = self.__nev_params('waveform_size')[channel_id]

            waveforms = spikes['waveform'].flatten().view(wf_dtype)
            waveforms = waveforms.reshape(int(spikes.size), 1, int(wf_size))

            if scaling == 'voltage':
                st.waveforms = (
                    waveforms[mask] * self.__nev_params('waveform_unit') *
                    self.__nev_params('digitization_factor')[channel_id] /
                    1000.)
            elif scaling == 'raw':
                st.waveforms = waveforms[mask] * pq.dimensionless
            else:
                raise ValueError(
                    'Unkown option {1} for parameter scaling.'.format(scaling))
            st.sampling_rate = self.__nev_params('waveform_sampling_rate')
            st.left_sweep = self.__get_left_sweep_waveforms()[channel_id]

        # add additional annotations
        st.annotate(
            unit_id=int(unit_id),
            channel_id=int(channel_id))

        return st

    def __read_analogsignal(
            self, n_start, n_stop, signal, channel_id, nsx_nb,
            scaling='raw', lazy=False):
        """
        Creates analogsignal for signal of channel in nsx data.
        """

        # TODO: The following part is extremely slow, since the memmaps for the
        # headers are created again and again. In particular, this makes lazy
        # loading slow as well. Solution would be to create header memmaps up
        # front.

        # get parameters
        sampling_rate = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'sampling_rate', nsx_nb)
        nsx_time_unit = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'time_unit', nsx_nb)
        max_ana = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'max_analog_val', nsx_nb)
        min_ana = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'min_analog_val', nsx_nb)
        max_dig = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'max_digital_val', nsx_nb)
        min_dig = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'min_digital_val', nsx_nb)
        units = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'units', nsx_nb)
        labels = self.__nsx_params[self.__nsx_spec[nsx_nb]](
            'labels', nsx_nb)

        dbl_idx = self.__nsx_databl_param[self.__nsx_spec[nsx_nb]](
            'databl_idx', nsx_nb, n_start, n_stop)
        t_start = self.__nsx_databl_param[self.__nsx_spec[nsx_nb]](
            'databl_t_start', nsx_nb, n_start, n_stop)
        t_stop = self.__nsx_databl_param[self.__nsx_spec[nsx_nb]](
            'databl_t_stop', nsx_nb, n_start, n_stop)

        elids_nsx = list(self.__nsx_ext_header[nsx_nb]['electrode_id'])
        if channel_id in elids_nsx:
            idx_ch = elids_nsx.index(channel_id)
        else:
            return None

        description = \
            "AnalogSignal from channel: {0}, label: {1}, nsx: {2}".format(
                channel_id, labels[idx_ch], nsx_nb)

        # TODO: Find a more time/memory efficient way to handle lazy loading
        data_times = np.arange(
            t_start.item(), t_stop.item(),
            self.__nsx_basic_header[nsx_nb]['period']) * t_start.units
        mask = (data_times >= n_start) & (data_times < n_stop)

        if lazy:
            lazy_shape = (np.sum(mask),)
            sig_ch = np.array([], dtype='float32')
            sig_unit = pq.dimensionless
            t_start = n_start.rescale('s')
        else:
            data_times = data_times[mask].astype(float)
            if scaling == 'voltage':
                if not self._avail_files['nev']:
                    raise ValueError(
                        'Cannot convert signals in filespec 2.1 nsX '
                        'files to voltage without nev file.')
                sig_ch = signal[dbl_idx][:, idx_ch][mask].astype('float32')

                # transform dig value to physical value
                sym_ana = (max_ana[idx_ch] == -min_ana[idx_ch])
                sym_dig = (max_dig[idx_ch] == -min_dig[idx_ch])
                if sym_ana and sym_dig:
                    sig_ch *= float(max_ana[idx_ch]) / float(max_dig[idx_ch])
                else:
                    # general case (same result as above for symmetric input)
                    sig_ch -= min_dig[idx_ch]
                    sig_ch *= float(max_ana[idx_ch] - min_ana[idx_ch]) / \
                              float(max_dig[idx_ch] - min_dig[idx_ch])
                    sig_ch += float(min_ana[idx_ch])
                sig_unit = units[idx_ch].decode()
            elif scaling == 'raw':
                sig_ch = signal[dbl_idx][:, idx_ch][mask].astype(int)
                sig_unit = pq.dimensionless
            else:
                raise ValueError(
                    'Unkown option {1} for parameter '
                    'scaling.'.format(scaling))

            t_start = data_times[0].rescale(nsx_time_unit)

        anasig = AnalogSignal(
            signal=pq.Quantity(sig_ch, sig_unit, copy=False),
            sampling_rate=sampling_rate,
            t_start=t_start,
            name=labels[idx_ch],
            description=description,
            file_origin='.'.join([self._filenames['nsx'], 'ns%i' % nsx_nb]))
        if lazy:
            anasig.lazy_shape = lazy_shape
        anasig.annotate(
            nsx=nsx_nb,
            channel_id=int(channel_id))
        return anasig

    def __read_unit(self, unit_id, channel_id):
        """
        Creates unit with unit id for given channel id.
        """
        # define a name for spiketrain
        # (unique identifier: 1000 * elid + unit_nb)
        name = "Unit {0}".format(1000 * channel_id + unit_id)
        # define description for spiketrain
        desc = 'Unit from channel: {0}, id: {1}'.format(
            channel_id, self.__get_unit_classification(unit_id))

        un = Unit(
            name=name,
            description=desc,
            file_origin='.'.join([self._filenames['nev'], 'nev']))

        # add additional annotations
        un.annotate(
            unit_id=int(unit_id),
            channel_id=int(channel_id))

        return un

    def __read_channelindex(
            self, channel_id, index=None, channel_units=None, cascade=True):
        """
        Returns a ChannelIndex with the given index for the given channels
        containing a neo.core.unit.Unit object list of the given units.
        """

        flt_type = {0: 'None', 1: 'Butterworth'}

        chidx = ChannelIndex(
            np.array([channel_id]),
            file_origin=self.filename)

        if index is not None:
            chidx.index = index
            chidx.name = "ChannelIndex {0}".format(chidx.index)
        else:
            chidx.name = "ChannelIndex"

        if self._avail_files['nev']:
            channel_labels = self.__nev_params('channel_labels')
            if channel_labels is not None:
                chidx.channel_names = np.array([channel_labels[channel_id]])
            chidx.channel_ids = np.array([channel_id])

            # additional annotations from nev
            if channel_id in self.__nev_ext_header[b'NEUEVWAV']['electrode_id']:
                get_idx = list(
                    self.__nev_ext_header[b'NEUEVWAV']['electrode_id']).index(
                    channel_id)
                chidx.annotate(
                    connector_ID=self.__nev_ext_header[
                        b'NEUEVWAV']['physical_connector'][get_idx],
                    connector_pinID=self.__nev_ext_header[
                        b'NEUEVWAV']['connector_pin'][get_idx],
                    nev_dig_factor=self.__nev_ext_header[
                        b'NEUEVWAV']['digitization_factor'][get_idx],
                    nev_energy_threshold=self.__nev_ext_header[
                                             b'NEUEVWAV']['energy_threshold'][get_idx] * pq.uV,
                    nev_hi_threshold=self.__nev_ext_header[
                                         b'NEUEVWAV']['hi_threshold'][get_idx] * pq.uV,
                    nev_lo_threshold=self.__nev_ext_header[
                                         b'NEUEVWAV']['lo_threshold'][get_idx] * pq.uV,
                    nb_sorted_units=self.__nev_ext_header[
                        b'NEUEVWAV']['nb_sorted_units'][get_idx],
                    waveform_size=self.__waveform_size[self.__nev_spec](
                    )[channel_id] * self.__nev_params('waveform_time_unit'))

                # additional annotations from nev (only for file_spec > 2.1)
                if self.__nev_spec in ['2.2', '2.3']:
                    get_idx = list(
                        self.__nev_ext_header[
                            b'NEUEVFLT']['electrode_id']).index(
                        channel_id)
                    # filter type codes (extracted from blackrock manual)
                    chidx.annotate(
                        nev_hi_freq_corner=self.__nev_ext_header[b'NEUEVFLT'][
                                               'hi_freq_corner'][get_idx] /
                                           1000. * pq.Hz,
                        nev_hi_freq_order=self.__nev_ext_header[b'NEUEVFLT'][
                            'hi_freq_order'][get_idx],
                        nev_hi_freq_type=flt_type[self.__nev_ext_header[
                            b'NEUEVFLT']['hi_freq_type'][get_idx]],
                        nev_lo_freq_corner=self.__nev_ext_header[
                                               b'NEUEVFLT']['lo_freq_corner'][get_idx] /
                                           1000. * pq.Hz,
                        nev_lo_freq_order=self.__nev_ext_header[
                            b'NEUEVFLT']['lo_freq_order'][get_idx],
                        nev_lo_freq_type=flt_type[self.__nev_ext_header[
                            b'NEUEVFLT']['lo_freq_type'][get_idx]])

        # additional information about the LFP signal
        if self.__nev_spec in ['2.2', '2.3'] and self.__nsx_ext_header:
            # It does not matter which nsX file to ask for this info
            k = list(self.__nsx_ext_header.keys())[0]
            if channel_id in self.__nsx_ext_header[k]['electrode_id']:
                get_idx = list(
                    self.__nsx_ext_header[k]['electrode_id']).index(
                    channel_id)
                chidx.annotate(
                    nsx_hi_freq_corner=self.__nsx_ext_header[k][
                                           'hi_freq_corner'][get_idx] / 1000. * pq.Hz,
                    nsx_lo_freq_corner=self.__nsx_ext_header[k][
                                           'lo_freq_corner'][get_idx] / 1000. * pq.Hz,
                    nsx_hi_freq_order=self.__nsx_ext_header[k][
                        'hi_freq_order'][get_idx],
                    nsx_lo_freq_order=self.__nsx_ext_header[k][
                        'lo_freq_order'][get_idx],
                    nsx_hi_freq_type=flt_type[
                        self.__nsx_ext_header[k]['hi_freq_type'][get_idx]],
                    nsx_lo_freq_type=flt_type[
                        self.__nsx_ext_header[k]['hi_freq_type'][get_idx]])

        chidx.description = \
            "Container for units and groups analogsignals of one recording " \
            "channel across segments."

        if not cascade:
            return chidx

        if self._avail_files['nev']:
            # read nev data
            nev_data = self.__nev_data_reader[self.__nev_spec]()

            if channel_units is not None:
                # extract first data for channel
                ch_mask = (nev_data['Spikes']['packet_id'] == channel_id)
                data_ch = nev_data['Spikes'][ch_mask]

                for un_id in channel_units:
                    if un_id in np.unique(data_ch['unit_class_nb']):
                        un = self.__read_unit(
                            unit_id=un_id, channel_id=channel_id)

                        chidx.units.append(un)

        chidx.create_many_to_one_relationship()

        return chidx

    def read_segment(
            self, n_start, n_stop, name=None, description=None, index=None,
            nsx_to_load='none', channels='none', units='none',
            load_waveforms=False, load_events=False, scaling='raw',
            lazy=False, cascade=True):
        """
        Returns an annotated neo.core.segment.Segment.

        Args:
            n_start (Quantity):
                Start time of maximum time range of signals contained in this
                segment.
            n_stop (Quantity):
                Stop time of maximum time range of signals contained in this
                segment.
            name (None, string):
                If None, name is set to default, otherwise it is set to user
                input.
            description (None, string):
                If None, description is set to default, otherwise it is set to
                user input.
            index (None, int):
                If not None, index of segment is set to user index.
            nsx_to_load (int, list, str):
                ID(s) of nsx file(s) from which to load data, e.g., if set to
                5 only data from the ns5 file are loaded. If 'none' or empty
                list, no nsx files and therefore no analog signals are loaded.
                If 'all', data from all available nsx are loaded.
            channels (int, list, str):
                Channel id(s) from which to load data. If 'none' or empty list,
                no channels and therefore no analog signal or spiketrains are
                loaded. If 'all', all available channels are loaded.
            units (int, list, str, dict):
                ID(s) of unit(s) to load. If 'none' or empty list, no units and
                therefore no spiketrains are loaded. If 'all', all available
                units are loaded. If dict, the above can be specified
                individually for each channel (keys), e.g. {1: 5, 2: 'all'}
                loads unit 5 from channel 1 and all units from channel 2.
            load_waveforms (boolean):
                If True, waveforms are attached to all loaded spiketrains.
            load_events (boolean):
                If True, all recorded events are loaded.
            scaling (str):
                Determines whether time series of individual
                electrodes/channels are returned as AnalogSignals containing
                raw integer samples ('raw'), or scaled to arrays of floats
                representing voltage ('voltage'). Note that for file
                specification 2.1 and lower, the option 'voltage' requires a
                nev file to be present.
            lazy (boolean):
                If True, only the shape of the data is loaded.
            cascade (boolean):
                If True, only the segment without children is returned.

        Returns:
            Segment (neo.Segment):
                Returns the specified segment. See documentation of
                `read_block()` for a full list of annotations of all child
                objects.
        """

        # Make sure that input args are transformed into correct instances
        nsx_to_load = self.__transform_nsx_to_load(nsx_to_load)
        channels = self.__transform_channels(channels, nsx_to_load)
        units = self.__transform_units(units, channels)

        seg = Segment(file_origin=self.filename)

        # set user defined annotations if they were provided
        if index is None:
            seg.index = 0
        else:
            seg.index = index
        if name is None:
            seg.name = "Segment {0}".format(seg.index)
        else:
            seg.name = name
        if description is None:
            seg.description = "Segment containing data from t_min to t_max."
        else:
            seg.description = description

        if not cascade:
            return seg

        if self._avail_files['nev']:
            #            filename = self._filenames['nev'] + '.nev'
            # annotate segment according to file headers
            seg.rec_datetime = datetime.datetime(
                year=self.__nev_basic_header['year'],
                month=self.__nev_basic_header['month'],
                day=self.__nev_basic_header['day'],
                hour=self.__nev_basic_header['hour'],
                minute=self.__nev_basic_header['minute'],
                second=self.__nev_basic_header['second'],
                microsecond=self.__nev_basic_header['millisecond'])

            # read nev data
            nev_data = self.__nev_data_reader[self.__nev_spec]()

            # read non-neural experimental events
            if load_events:
                ev_dict = self.__nonneural_evtypes[self.__nev_spec](
                    nev_data['NonNeural'])

                for ev_type in ev_dict.keys():

                    ev = self.__read_event(
                        n_start=n_start,
                        n_stop=n_stop,
                        data=nev_data['NonNeural'],
                        ev_dict=ev_dict[ev_type],
                        lazy=lazy)

                    if ev is not None:
                        seg.events.append(ev)

                        # TODO: not yet implemented (only avail in nev_spec 2.3)
                        # videosync events
                        # trackingevents events
                        # buttontrigger events
                        # configevent events

            # get spiketrain
            if units is not None:
                not_existing_units = []
                for ch_id in units.keys():
                    # extract first data for channel
                    ch_mask = (nev_data['Spikes']['packet_id'] == ch_id)
                    data_ch = nev_data['Spikes'][ch_mask]
                    if units[ch_id] is not None:
                        for un_id in units[ch_id]:
                            if un_id in np.unique(data_ch['unit_class_nb']):
                                # extract then data for unit if unit exists
                                un_mask = (data_ch['unit_class_nb'] == un_id)
                                data_un = data_ch[un_mask]

                                st = self.__read_spiketrain(
                                    n_start=n_start,
                                    n_stop=n_stop,
                                    spikes=data_un,
                                    channel_id=ch_id,
                                    unit_id=un_id,
                                    load_waveforms=load_waveforms,
                                    scaling=scaling,
                                    lazy=lazy)

                                seg.spiketrains.append(st)
                            else:
                                not_existing_units.append(un_id)

                        if not_existing_units:
                            self._print_verbose(
                                "Units {0} on channel {1} do not "
                                "exist".format(not_existing_units, ch_id))
                    else:
                        self._print_verbose(
                            "There are no units specified for channel "
                            "{0}".format(ch_id))

        if nsx_to_load is not None:
            for nsx_nb in nsx_to_load:
                # read nsx data
                nsx_data = \
                    self.__nsx_data_reader[self.__nsx_spec[nsx_nb]](nsx_nb)

                # read Analogsignals
                for ch_id in channels:

                    anasig = self.__read_analogsignal(
                        n_start=n_start,
                        n_stop=n_stop,
                        signal=nsx_data,
                        channel_id=ch_id,
                        nsx_nb=nsx_nb,
                        scaling=scaling,
                        lazy=lazy)

                    if anasig is not None:
                        seg.analogsignals.append(anasig)

                        # TODO: not yet implemented
                    #        if self._avail_files['sif']:
                    #            sif_header = self._read_sif(self._filenames['sif'] + '.sif')

                    # TODO: not yet implemented
                    #        if self._avail_files['ccf']:
                    #            ccf_header = self._read_sif(self._filenames['ccf'] + '.ccf')

        seg.create_many_to_one_relationship()

        return seg

    def read_block(
            self, index=None, name=None, description=None, nsx_to_load='none',
            n_starts=None, n_stops=None, channels='none', units='none',
            load_waveforms=False, load_events=False, scaling='raw',
            lazy=False, cascade=True):
        """
        Args:
            index (None, int):
                If not None, index of block is set to user input.
            name (None, str):
                If None, name is set to default, otherwise it is set to user
                input.
            description (None, str):
                If None, description is set to default, otherwise it is set to
                user input.
            nsx_to_load (int, list, str):
                ID(s) of nsx file(s) from which to load data, e.g., if set to
                5 only data from the ns5 file are loaded. If 'none' or empty
                list, no nsx files and therefore no analog signals are loaded.
                If 'all', data from all available nsx are loaded.
            n_starts (None, Quantity, list):
                Start times for data in each segment. Number of entries must be
                equal to length of n_stops. If None, intrinsic recording start
                times of files set are used.
            n_stops (None, Quantity, list):
                Stop times for data in each segment. Number of entries must be
                equal to length of n_starts. If None, intrinsic recording stop
                times of files set are used.
            channels (int, list, str):
                Channel id(s) from which to load data. If 'none' or empty list,
                no channels and therefore no analog signal or spiketrains are
                loaded. If 'all', all available channels are loaded.
            units (int, list, str, dict):
                ID(s) of unit(s) to load. If 'none' or empty list, no units and
                therefore no spiketrains are loaded. If 'all', all available
                units are loaded. If dict, the above can be specified
                individually for each channel (keys), e.g. {1: 5, 2: 'all'}
                loads unit 5 from channel 1 and all units from channel 2.
            load_waveforms (boolean):
                If True, waveforms are attached to all loaded spiketrains.
            load_events (boolean):
                If True, all recorded events are loaded.
            scaling (str):
                Determines whether time series of individual
                electrodes/channels are returned as AnalogSignals containing
                raw integer samples ('raw'), or scaled to arrays of floats
                representing voltage ('voltage'). Note that for file
                specification 2.1 and lower, the option 'voltage' requires a
                nev file to be present.
            lazy (bool):
                If True, only the shape of the data is loaded.
            cascade (bool or "lazy"):
                If True, only the block without children is returned.

        Returns:
            Block (neo.segment.Block):
                Block linking all loaded Neo objects.

                Block annotations:
                    avail_file_set (list):
                        List of extensions of all available files for the given
                        recording.
                    avail_nsx (list of int):
                        List of integers specifying the .nsX files available,
                        e.g., [2, 5] indicates that an ns2 and and ns5 file are
                        available.
                    avail_nev (bool):
                        True if a .nev file is available.
                    avail_ccf (bool):
                        True if a .ccf file is available.
                    avail_sif (bool):
                        True if a .sif file is available.
                    rec_pauses (bool):
                        True if the session contains a recording pause (i.e.,
                        multiple segments).
                    nb_segments (int):
                        Number of segments created after merging recording
                        times specified by user with the intrinsic ones of the
                        file set.

                Segment annotations:
                    None.

                ChannelIndex annotations:
                    waveform_size (Quantitiy):
                        Length of time used to save spike waveforms (in units
                        of 1/30000 s).
                    nev_hi_freq_corner (Quantitiy),
                    nev_lo_freq_corner (Quantitiy),
                    nev_hi_freq_order (int), nev_lo_freq_order (int),
                    nev_hi_freq_type (str), nev_lo_freq_type (str),
                    nev_hi_threshold, nev_lo_threshold,
                    nev_energy_threshold (quantity):
                        Indicates parameters of spike detection.
                    nsx_hi_freq_corner (Quantity),
                    nsx_lo_freq_corner (Quantity)
                    nsx_hi_freq_order (int), nsx_lo_freq_order (int),
                    nsx_hi_freq_type (str), nsx_lo_freq_type (str)
                        Indicates parameters of the filtered signal in one of
                        the files ns1-ns5 (ns6, if available, is not filtered).
                    nev_dig_factor (int):
                        Digitization factor in microvolts of the nev file, used
                        to convert raw samples to volt.
                    connector_ID, connector_pinID (int):
                        ID of connector and pin on the connector where the
                        channel was recorded from.
                    nb_sorted_units (int):
                        Number of sorted units on this channel (noise, mua and
                        sua).

                Unit annotations:
                    unit_id (int):
                        ID of the unit.
                    channel_id (int):
                        Channel ID (Blackrock ID) from which the unit was
                        loaded (equiv. to the single list entry in the
                        attribute channel_ids of ChannelIndex parent).

                AnalogSignal annotations:
                    nsx (int):
                        nsX file the signal was loaded from, e.g., 5 indicates
                        the .ns5 file.
                    channel_id (int):
                        Channel ID (Blackrock ID) from which the signal was
                        loaded.

                Spiketrain annotations:
                    unit_id (int):
                        ID of the unit from which the spikes were recorded.
                    channel_id (int):
                        Channel ID (Blackrock ID) from which the spikes were
                        loaded.

                Event annotations:
                    The resulting Block contains one Event object with the name
                    `digital_input_port`. It contains all digitally recorded
                    events, with the event code coded in the labels of the
                    Event. The Event object contains no further annotation.
        """
        # Make sure that input args are transformed into correct instances
        nsx_to_load = self.__transform_nsx_to_load(nsx_to_load)
        channels = self.__transform_channels(channels, nsx_to_load)
        units = self.__transform_units(units, channels)

        # Create block
        bl = Block(file_origin=self.filename)

        # set user defined annotations if they were provided
        if index is not None:
            bl.index = index
        if name is None:
            bl.name = "Blackrock Data Block"
        else:
            bl.name = name
        if description is None:
            bl.description = "Block of data from Blackrock file set."
        else:
            bl.description = description

        if self._avail_files['nev']:
            bl.rec_datetime = self.__nev_params('rec_datetime')

        bl.annotate(
            avail_file_set=[k for k, v in self._avail_files.items() if v])
        bl.annotate(avail_nsx=self._avail_nsx)
        bl.annotate(avail_nev=self._avail_files['nev'])
        bl.annotate(avail_sif=self._avail_files['sif'])
        bl.annotate(avail_ccf=self._avail_files['ccf'])
        bl.annotate(rec_pauses=False)

        # Test n_starts and n_stops user requirements and combine them if
        # possible with file internal n_starts and n_stops from rec pauses.
        n_starts, n_stops = \
            self.__merge_time_ranges(n_starts, n_stops, nsx_to_load)

        bl.annotate(nb_segments=len(n_starts))

        if not cascade:
            return bl

        # read segment
        for seg_idx, (n_start, n_stop) in enumerate(zip(n_starts, n_stops)):
            seg = self.read_segment(
                n_start=n_start,
                n_stop=n_stop,
                index=seg_idx,
                nsx_to_load=nsx_to_load,
                channels=channels,
                units=units,
                load_waveforms=load_waveforms,
                load_events=load_events,
                scaling=scaling,
                lazy=lazy,
                cascade=cascade)

            bl.segments.append(seg)

        # read channelindexes
        if channels:
            for ch_id in channels:
                if units and ch_id in units.keys():
                    ch_units = units[ch_id]
                else:
                    ch_units = None

                chidx = self.__read_channelindex(
                    channel_id=ch_id,
                    index=0,
                    channel_units=ch_units,
                    cascade=cascade)

                for seg in bl.segments:
                    if ch_units:
                        for un in chidx.units:
                            sts = seg.filter(
                                targdict={'name': un.name},
                                objects='SpikeTrain')
                            for st in sts:
                                un.spiketrains.append(st)

                    anasigs = seg.filter(
                        targdict={'channel_id': ch_id},
                        objects='AnalogSignal')
                    for anasig in anasigs:
                        chidx.analogsignals.append(anasig)

                bl.channel_indexes.append(chidx)

        bl.create_many_to_one_relationship()

        return bl

    def __str__(self):
        """
        Prints summary of the Blackrock data file set.
        """
        output = "\nFile Origins for Blackrock File Set\n" \
                 "====================================\n"
        for ftype in self._filenames.keys():
            output += ftype + ':' + self._filenames[ftype] + '\n'

        if self._avail_files['nev']:
            output += "\nEvent Parameters (NEV)\n" \
                      "====================================\n" \
                      "Timestamp resolution (Hz): " + \
                      str(self.__nev_basic_header['timestamp_resolution']) + \
                      "\nWaveform resolution (Hz): " + \
                      str(self.__nev_basic_header['sample_resolution'])

            if b'NEUEVWAV' in self.__nev_ext_header.keys():
                avail_el = \
                    self.__nev_ext_header[b'NEUEVWAV']['electrode_id']
                con = \
                    self.__nev_ext_header[b'NEUEVWAV']['physical_connector']
                pin = \
                    self.__nev_ext_header[b'NEUEVWAV']['connector_pin']
                nb_units = \
                    self.__nev_ext_header[b'NEUEVWAV']['nb_sorted_units']
                output += "\n\nAvailable electrode IDs:\n" \
                          "====================================\n"
                for i, el in enumerate(avail_el):
                    output += "Electrode ID %i: " % el

                    channel_labels = self.__nev_params('channel_labels')
                    if channel_labels is not None:
                        output += "label %s: " % channel_labels[el]

                    output += "connector: %i, " % con[i]
                    output += "pin: %i, " % pin[i]
                    output += 'nb_units: %i\n' % nb_units[i]
        for nsx_nb in self._avail_nsx:
            analog_res = self.__nsx_params[self.__nsx_spec[nsx_nb]](
                'sampling_rate', nsx_nb)
            avail_el = [
                el for el in self.__nsx_ext_header[nsx_nb]['electrode_id']]
            output += "\nAnalog Parameters (NS" + str(nsx_nb) + ")" \
                                                                "\n===================================="
            output += "\nResolution (Hz): %i" % analog_res
            output += "\nAvailable channel IDs: " + \
                      ", ".join(["%i" % a for a in avail_el]) + "\n"

        return output
