# -*- coding: utf-8 -*-
"""

Classe for reading data from pCLAMP and AxoScope
files (.abf version 1 and 2), developed by Molecular device/Axon technologies.

- abf = Axon binary file
- atf is a text file based format from axon that could be
  read by AsciiIO (but this file is less efficient.)


This code is a port of abfload and abf2load
written in Matlab (BSD-2-Clause licence) by :
 - Copyright (c) 2009, Forrest Collman, fcollman@princeton.edu
 - Copyright (c) 2004, Harald Hentschke
and available here :
http://www.mathworks.com/matlabcentral/fileexchange/22114-abf2load

Information on abf 1 and 2 formats is available here :
http://www.moleculardevices.com/pages/software/developer_info.html

This file supports the old (ABF1) and new (ABF2) format.
ABF1 (clampfit <=9) and ABF2 (clampfit >10)

All possible mode are possible :
    - event-driven variable-length mode 1 -> return several Segments per Block
    - event-driven fixed-length mode 2 or 5 -> return several Segments
    - gap free mode -> return one (or sevral) Segment in the Block

Supported : Read

Author: sgarcia, jnowacki

Note: j.s.nowacki@gmail.com has a C++ library with SWIG bindings which also
reads abf files - would be good to cross-check

"""

import struct
import datetime
import os
from io import open, BufferedReader

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import *
from neo.io.tools import iteritems


class struct_file(BufferedReader):
    def read_f(self, fmt, offset=None):
        if offset is not None:
            self.seek(offset)
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

    def write_f(self, fmt, offset=None, *args):
        if offset is not None:
            self.seek(offset)
        self.write(struct.pack(fmt, *args))


def reformat_integer_V1(data, nbchannel, header):
    """
    reformat when dtype is int16 for ABF version 1
    """
    chans = [chan_num for chan_num in
             header['nADCSamplingSeq'] if chan_num >= 0]
    for n, i in enumerate(chans[:nbchannel]):  # respect SamplingSeq
        data[:, n] /= header['fInstrumentScaleFactor'][i]
        data[:, n] /= header['fSignalGain'][i]
        data[:, n] /= header['fADCProgrammableGain'][i]
        if header['nTelegraphEnable'][i]:
            data[:, n] /= header['fTelegraphAdditGain'][i]
        data[:, n] *= header['fADCRange']
        data[:, n] /= header['lADCResolution']
        data[:, n] += header['fInstrumentOffset'][i]
        data[:, n] -= header['fSignalOffset'][i]


def reformat_integer_V2(data, nbchannel, header):
    """
    reformat when dtype is int16 for ABF version 2
    """
    for i in range(nbchannel):
        data[:, i] /= header['listADCInfo'][i]['fInstrumentScaleFactor']
        data[:, i] /= header['listADCInfo'][i]['fSignalGain']
        data[:, i] /= header['listADCInfo'][i]['fADCProgrammableGain']
        if header['listADCInfo'][i]['nTelegraphEnable']:
            data[:, i] /= header['listADCInfo'][i]['fTelegraphAdditGain']
        data[:, i] *= header['protocol']['fADCRange']
        data[:, i] /= header['protocol']['lADCResolution']
        data[:, i] += header['listADCInfo'][i]['fInstrumentOffset']
        data[:, i] -= header['listADCInfo'][i]['fSignalOffset']


def clean_string(s):
    s = s.rstrip(b'\x00')
    s = s.rstrip(b' ')
    return s


class AxonIO(BaseIO):
    """

    Class for reading abf (axon binary file) file.

    Usage:
        >>> from neo import io
        >>> r = io.AxonIO(filename='File_axon_1.abf')
        >>> bl = r.read_block(lazy=False, cascade=True)
        >>> print bl.segments
        [<neo.core.segment.Segment object at 0x105516fd0>]
        >>> print bl.segments[0].analogsignals
        [<AnalogSignal(array([2.18811035, 2.19726562, 2.21252441, ...,
        1.33056641, 1.3458252,  1.3671875], dtype=float32) * pA,
        [0.0 s, 191.2832 s], sampling rate: 10000.0 Hz)>]
        >>> print bl.segments[0].eventarrays
        []

    """

    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal, EventArray]
    readable_objects = [Block]
    writeable_objects = []

    has_header = False
    is_streameable = False

    read_params = {Block: []}
    write_params = None

    name = 'Axon'
    extensions = ['abf']

    mode = 'file'

    def __init__(self, filename=None):
        """
        This class read a abf file.

        Arguments:
            filename : the filename to read

        """
        BaseIO.__init__(self)
        self.filename = filename

    def read_block(self, lazy=False, cascade=True):

        header = self.read_header()
        version = header['fFileVersionNumber']

        bl = Block()
        bl.file_origin = os.path.basename(self.filename)
        bl.annotate(abf_version=version)

        # date and time
        if version < 2.:
            YY = 1900
            MM = 1
            DD = 1
            hh = int(header['lFileStartTime'] / 3600.)
            mm = int((header['lFileStartTime'] - hh * 3600) / 60)
            ss = header['lFileStartTime'] - hh * 3600 - mm * 60
            ms = int(np.mod(ss, 1) * 1e6)
            ss = int(ss)
        elif version >= 2.:
            YY = int(header['uFileStartDate'] / 10000)
            MM = int((header['uFileStartDate'] - YY * 10000) / 100)
            DD = int(header['uFileStartDate'] - YY * 10000 - MM * 100)
            hh = int(header['uFileStartTimeMS'] / 1000. / 3600.)
            mm = int((header['uFileStartTimeMS'] / 1000. - hh * 3600) / 60)
            ss = header['uFileStartTimeMS'] / 1000. - hh * 3600 - mm * 60
            ms = int(np.mod(ss, 1) * 1e6)
            ss = int(ss)
        bl.rec_datetime = datetime.datetime(YY, MM, DD, hh, mm, ss, ms)

        if not cascade:
            return bl

        # file format
        if header['nDataFormat'] == 0:
            dt = np.dtype('i2')
        elif header['nDataFormat'] == 1:
            dt = np.dtype('f4')

        if version < 2.:
            nbchannel = header['nADCNumChannels']
            headOffset = header['lDataSectionPtr'] * BLOCKSIZE +\
                header['nNumPointsIgnored'] * dt.itemsize
            totalsize = header['lActualAcqLength']
        elif version >= 2.:
            nbchannel = header['sections']['ADCSection']['llNumEntries']
            headOffset = header['sections']['DataSection']['uBlockIndex'] *\
                BLOCKSIZE
            totalsize = header['sections']['DataSection']['llNumEntries']

        data = np.memmap(self.filename, dt, 'r',
                         shape=(totalsize,), offset=headOffset)

        # 3 possible modes
        if version < 2.:
            mode = header['nOperationMode']
        elif version >= 2.:
            mode = header['protocol']['nOperationMode']

        if (mode == 1) or (mode == 2) or (mode == 5) or (mode == 3):
            # event-driven variable-length mode (mode 1)
            # event-driven fixed-length mode (mode 2 or 5)
            # gap free mode 3 can be in several episod (strange but possible)

            # read sweep pos
            if version < 2.:
                nbepisod = header['lSynchArraySize']
                offsetEpisod = header['lSynchArrayPtr'] * BLOCKSIZE
            elif version >= 2.:
                SAS = header['sections']['SynchArraySection']
                nbepisod = SAS['llNumEntries']
                offsetEpisod = SAS['uBlockIndex'] * BLOCKSIZE
            if nbepisod > 0:
                episodArray = np.memmap(self.filename, [('offset', 'i4'),
                                        ('len', 'i4')], 'r', shape=(nbepisod),
                                        offset=offsetEpisod)
            else:
                episodArray = np.empty((1), [('offset', 'i4'), ('len', 'i4')],)
                episodArray[0]['len'] = data.size
                episodArray[0]['offset'] = 0

            # sampling_rate
            if version < 2.:
                sampling_rate = 1. / (header['fADCSampleInterval'] *
                                      nbchannel * 1.e-6) * pq.Hz
            elif version >= 2.:
                sampling_rate = 1.e6 / \
                    header['protocol']['fADCSequenceInterval'] * pq.Hz

            # construct block
            # one sweep = one segment in a block
            pos = 0
            for j in range(episodArray.size):
                seg = Segment(index=j)

                length = episodArray[j]['len']

                if version < 2.:
                    fSynchTimeUnit = header['fSynchTimeUnit']
                elif version >= 2.:
                    fSynchTimeUnit = header['protocol']['fSynchTimeUnit']

                if (fSynchTimeUnit != 0) and (mode == 1):
                    length /= fSynchTimeUnit

                if not lazy:
                    subdata = data[pos:pos+length]
                    subdata = subdata.reshape((subdata.size/nbchannel,
                                               nbchannel)).astype('f')
                    if dt == np.dtype('i2'):
                        if version < 2.:
                            reformat_integer_V1(subdata, nbchannel, header)
                        elif version >= 2.:
                            reformat_integer_V2(subdata, nbchannel, header)

                pos += length

                if version < 2.:
                    chans = [chan_num for chan_num in
                             header['nADCSamplingSeq'] if chan_num >= 0]
                else:
                    chans = range(nbchannel)
                for n, i in enumerate(chans[:nbchannel]):  # fix SamplingSeq
                    if version < 2.:
                        name = header['sADCChannelName'][i].replace(b' ', b'')
                        unit = header['sADCUnits'][i].replace(b'\xb5', b'u').\
                            replace(b' ', b'').decode('utf-8')  # \xb5 is µ
                        num = header['nADCPtoLChannelMap'][i]
                    elif version >= 2.:
                        lADCIi = header['listADCInfo'][i]
                        name = lADCIi['ADCChNames'].replace(b' ', b'')
                        unit = lADCIi['ADCChUnits'].replace(b'\xb5', b'u').\
                            replace(b' ', b'').decode('utf-8')
                        num = header['listADCInfo'][i]['nADCNum']
                    if (fSynchTimeUnit == 0):
                        t_start = float(episodArray[j]['offset']) / sampling_rate
                    else:
                        t_start = float(episodArray[j]['offset']) * fSynchTimeUnit *1e-6* pq.s
                    t_start = t_start.rescale('s')
                    try:
                        pq.Quantity(1, unit)
                    except:
                        unit = ''

                    if lazy:
                        signal = [] * pq.Quantity(1, unit)
                    else:
                        signal = pq.Quantity(subdata[:, n], unit)

                    anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                          t_start=t_start,
                                          name=str(name),
                                          channel_index=int(num))
                    if lazy:
                        anaSig.lazy_shape = length / nbchannel
                    seg.analogsignals.append(anaSig)
                bl.segments.append(seg)

            if mode in [3, 5]:  # TODO check if tags exits in other mode

                # tag is EventArray that should be attached to Block
                # It is attched to the first Segment
                times = []
                labels = []
                comments = []
                for i, tag in enumerate(header['listTag']):
                    times.append(tag['lTagTime']/sampling_rate)
                    labels.append(str(tag['nTagType']))
                    comments.append(clean_string(tag['sComment']))
                times = np.array(times)
                labels = np.array(labels, dtype='S')
                comments = np.array(comments, dtype='S')
                # attach all tags to the first segment.
                seg = bl.segments[0]
                if lazy:
                    ea = EventArray(times=[] * pq.s,
                                    labels=np.array([], dtype='S'))
                    ea.lazy_shape = len(times)
                else:
                    ea = EventArray(times=times*pq.s,
                                    labels=labels, comments=comments)
                seg.eventarrays.append(ea)

        bl.create_many_to_one_relationship()
        return bl

    def read_header(self,):
        """
        read the header of the file

        The strategy differ here from the original script under Matlab.
        In the original script for ABF2, it complete the header with
        informations that are located in other structures.

        In ABF2 this function return header with sub dict :
            sections             (ABF2)
            protocol             (ABF2)
            listTags             (ABF1&2)
            listADCInfo          (ABF2)
            listDACInfo          (ABF2)
            dictEpochInfoPerDAC  (ABF2)
        that contain more information.
        """
        fid = struct_file(open(self.filename, 'rb'))  # fix for py3

        # version
        fFileSignature = fid.read(4)
        if fFileSignature == b'ABF ':  # fix for p3 where read returns bytes
            headerDescription = headerDescriptionV1
        elif fFileSignature == b'ABF2':
            headerDescription = headerDescriptionV2
        else:
            return None

        # construct dict
        header = {}
        for key, offset, fmt in headerDescription:
            val = fid.read_f(fmt, offset=offset)
            if len(val) == 1:
                header[key] = val[0]
            else:
                header[key] = np.array(val)

        # correction of version number and starttime
        if fFileSignature == b'ABF ':
            header['lFileStartTime'] = header['lFileStartTime'] +\
                header['nFileStartMillisecs'] * .001
        elif fFileSignature == b'ABF2':
            n = header['fFileVersionNumber']
            header['fFileVersionNumber'] = n[3] + 0.1 * n[2] +\
                0.01 * n[1] + 0.001 * n[0]
            header['lFileStartTime'] = header['uFileStartTimeMS'] * .001

        if header['fFileVersionNumber'] < 2.:
            # tags
            listTag = []
            for i in range(header['lNumTagEntries']):
                fid.seek(header['lTagSectionPtr'] + i * 64)
                tag = {}
                for key, fmt in TagInfoDescription:
                    val = fid.read_f(fmt)
                    if len(val) == 1:
                        tag[key] = val[0]
                    else:
                        tag[key] = np.array(val)
                listTag.append(tag)
            header['listTag'] = listTag
            #protocol name formatting #TODO move to read_protocol?
            header['sProtocolPath'] = clean_string(header['sProtocolPath'])
            header['sProtocolPath'] = header['sProtocolPath'].\
                replace(b'\\', b'/')

        elif header['fFileVersionNumber'] >= 2.:
            # in abf2 some info are in other place

            # sections
            sections = {}
            for s, sectionName in enumerate(sectionNames):
                uBlockIndex, uBytes, llNumEntries =\
                    fid.read_f('IIl', offset=76 + s * 16)
                sections[sectionName] = {}
                sections[sectionName]['uBlockIndex'] = uBlockIndex
                sections[sectionName]['uBytes'] = uBytes
                sections[sectionName]['llNumEntries'] = llNumEntries
            header['sections'] = sections

            # strings sections
            # hack for reading channels names and units
            fid.seek(sections['StringsSection']['uBlockIndex'] * BLOCKSIZE)
            bigString = fid.read(sections['StringsSection']['uBytes'])
            goodstart = bigString.lower().find(b'clampex')
            if goodstart == -1:
                goodstart = bigString.lower().find(b'axoscope')

            bigString = bigString[goodstart:]
            strings = bigString.split(b'\x00')

            # ADC sections
            header['listADCInfo'] = []
            for i in range(sections['ADCSection']['llNumEntries']):
                #  read ADCInfo
                fid.seek(sections['ADCSection']['uBlockIndex'] *
                         BLOCKSIZE + sections['ADCSection']['uBytes'] * i)
                ADCInfo = {}
                for key, fmt in ADCInfoDescription:
                    val = fid.read_f(fmt)
                    if len(val) == 1:
                        ADCInfo[key] = val[0]
                    else:
                        ADCInfo[key] = np.array(val)
                ADCInfo['ADCChNames'] = strings[ADCInfo['lADCChannelNameIndex']
                                                - 1]
                ADCInfo['ADCChUnits'] = strings[ADCInfo['lADCUnitsIndex'] - 1]

                header['listADCInfo'].append(ADCInfo)

            # protocol sections
            protocol = {}
            fid.seek(sections['ProtocolSection']['uBlockIndex'] * BLOCKSIZE)
            for key, fmt in protocolInfoDescription:
                val = fid.read_f(fmt)
                if len(val) == 1:
                    protocol[key] = val[0]
                else:
                    protocol[key] = np.array(val)
            header['protocol'] = protocol

            # tags
            listTag = []
            for i in range(sections['TagSection']['llNumEntries']):
                fid.seek(sections['TagSection']['uBlockIndex'] *
                         BLOCKSIZE + sections['TagSection']['uBytes'] * i)
                tag = {}
                for key, fmt in TagInfoDescription:
                    val = fid.read_f(fmt)
                    if len(val) == 1:
                        tag[key] = val[0]
                    else:
                        tag[key] = np.array(val)
                listTag.append(tag)

            header['listTag'] = listTag

            # DAC sections
            header['listDACInfo'] = []
            for i in range(sections['DACSection']['llNumEntries']):
                # read DACInfo
                fid.seek(sections['DACSection']['uBlockIndex'] *
                         BLOCKSIZE + sections['DACSection']['uBytes'] * i)
                DACInfo = {}
                for key, fmt in DACInfoDescription:
                    val = fid.read_f(fmt)
                    if len(val) == 1:
                        DACInfo[key] = val[0]
                    else:
                        DACInfo[key] = np.array(val)
                DACInfo['DACChNames'] = strings[DACInfo['lDACChannelNameIndex']
                                                - 1]
                DACInfo['DACChUnits'] = strings[
                    DACInfo['lDACChannelUnitsIndex'] - 1]

                header['listDACInfo'].append(DACInfo)

            # EpochPerDAC  sections
            # header['dictEpochInfoPerDAC'] is dict of dicts:
            #  - the first index is the DAC number
            #  - the second index is the epoch number
            # It has to be done like that because data may not exist
            # and may not be in sorted order
            header['dictEpochInfoPerDAC'] = {}
            for i in range(sections['EpochPerDACSection']['llNumEntries']):
                #  read DACInfo
                fid.seek(sections['EpochPerDACSection']['uBlockIndex'] *
                         BLOCKSIZE +
                         sections['EpochPerDACSection']['uBytes'] * i)
                EpochInfoPerDAC = {}
                for key, fmt in EpochInfoPerDACDescription:
                    val = fid.read_f(fmt)
                    if len(val) == 1:
                        EpochInfoPerDAC[key] = val[0]
                    else:
                        EpochInfoPerDAC[key] = np.array(val)

                DACNum = EpochInfoPerDAC['nDACNum']
                EpochNum = EpochInfoPerDAC['nEpochNum']
                # Checking if the key exists, if not, the value is empty
                # so we have to create empty dict to populate
                if DACNum not in header['dictEpochInfoPerDAC']:
                    header['dictEpochInfoPerDAC'][DACNum] = {}

                header['dictEpochInfoPerDAC'][DACNum][EpochNum] =\
                    EpochInfoPerDAC

        fid.close()

        return header

    def read_protocol(self):
        """
        Read the protocol waveform of the file, if present;
        function works with ABF2 only. Protocols can be reconstructed
        from the ABF1 header.

        Returns: list of segments (one for every episode)
                 with list of analog signls (one for every DAC).
        """
        header = self.read_header()

        if header['fFileVersionNumber'] < 2.:
            raise IOError("Protocol section is only present in ABF2 files.")

        nADC = header['sections']['ADCSection']['llNumEntries']  # n ADC chans
        nDAC = header['sections']['DACSection']['llNumEntries']  # n DAC chans
        nSam = header['protocol']['lNumSamplesPerEpisode']/nADC  # samples/ep
        nEpi = header['lActualEpisodes']
        sampling_rate = 1.e6/header['protocol']['fADCSequenceInterval'] * pq.Hz

        # Make a list of segments with analog signals with just holding levels
        # List of segments relates to number of episodes, as for recorded data
        segments = []
        for epiNum in range(nEpi):
            seg = Segment(index=epiNum)
            # One analog signal for each DAC in segment (episode)
            for DACNum in range(nDAC):
                t_start = 0 * pq.s  # TODO: Possibly check with episode array
                name = header['listDACInfo'][DACNum]['DACChNames']
                unit = header['listDACInfo'][DACNum]['DACChUnits'].\
                    replace(b'\xb5', b'u')  # \xb5 is µ
                signal = np.ones(nSam) *\
                    header['listDACInfo'][DACNum]['fDACHoldingLevel'] *\
                    pq.Quantity(1, unit)
                anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                      t_start=t_start, name=str(name),
                                      channel_index=DACNum)
                # If there are epoch infos for this DAC
                if DACNum in header['dictEpochInfoPerDAC']:
                    # Save last sample index
                    i_last = int(nSam * 15625 / 10**6)
                    # TODO guess for first holding
                    # Go over EpochInfoPerDAC and change the analog signal
                    # according to the epochs
                    epochInfo = header['dictEpochInfoPerDAC'][DACNum]
                    for epochNum, epoch in iteritems(epochInfo):
                        i_begin = i_last
                        i_end = i_last + epoch['lEpochInitDuration'] +\
                            epoch['lEpochDurationInc'] * epiNum
                        dif = i_end-i_begin
                        anaSig[i_begin:i_end] = np.ones(len(range(dif))) *\
                            pq.Quantity(1, unit) * (epoch['fEpochInitLevel'] +
                                                    epoch['fEpochLevelInc'] *
                                                    epiNum)
                        i_last += epoch['lEpochInitDuration']
                seg.analogsignals.append(anaSig)
            segments.append(seg)

        return segments

BLOCKSIZE = 512

headerDescriptionV1 = [
    ('fFileSignature', 0, '4s'),
    ('fFileVersionNumber', 4, 'f'),
    ('nOperationMode', 8, 'h'),
    ('lActualAcqLength', 10, 'i'),
    ('nNumPointsIgnored', 14, 'h'),
    ('lActualEpisodes', 16, 'i'),
    ('lFileStartTime', 24, 'i'),
    ('lDataSectionPtr', 40, 'i'),
    ('lTagSectionPtr', 44, 'i'),
    ('lNumTagEntries', 48, 'i'),
    ('lSynchArrayPtr', 92, 'i'),
    ('lSynchArraySize', 96, 'i'),
    ('nDataFormat', 100, 'h'),
    ('nADCNumChannels', 120, 'h'),
    ('fADCSampleInterval', 122, 'f'),
    ('fSynchTimeUnit', 130, 'f'),
    ('lNumSamplesPerEpisode', 138, 'i'),
    ('lPreTriggerSamples', 142, 'i'),
    ('lEpisodesPerRun', 146, 'i'),
    ('fADCRange', 244, 'f'),
    ('lADCResolution', 252, 'i'),
    ('nFileStartMillisecs', 366, 'h'),
    ('nADCPtoLChannelMap', 378, '16h'),
    ('nADCSamplingSeq', 410, '16h'),
    ('sADCChannelName', 442, '10s'*16),
    ('sADCUnits', 602, '8s'*16),
    ('fADCProgrammableGain', 730, '16f'),
    ('fInstrumentScaleFactor', 922, '16f'),
    ('fInstrumentOffset', 986, '16f'),
    ('fSignalGain', 1050, '16f'),
    ('fSignalOffset', 1114, '16f'),

    ('nDigitalEnable', 1436, 'h'),
    ('nActiveDACChannel', 1440, 'h'),
    ('nDigitalHolding', 1584, 'h'),
    ('nDigitalInterEpisode', 1586, 'h'),
    ('nDigitalValue', 2588, '10h'),
    ('lDACFilePtr', 2048, '2i'),
    ('lDACFileNumEpisodes', 2056, '2i'),
    ('fDACCalibrationFactor', 2074, '4f'),
    ('fDACCalibrationOffset', 2090, '4f'),
    ('nWaveformEnable', 2296, '2h'),
    ('nWaveformSource', 2300, '2h'),
    ('nInterEpisodeLevel', 2304, '2h'),
    ('nEpochType', 2308, '20h'),
    ('fEpochInitLevel', 2348, '20f'),
    ('fEpochLevelInc', 2428, '20f'),
    ('lEpochInitDuration', 2508, '20i'),
    ('lEpochDurationInc', 2588, '20i'),

    ('nTelegraphEnable', 4512, '16h'),
    ('fTelegraphAdditGain', 4576, '16f'),
    ('sProtocolPath', 4898, '384s'),
    ]


headerDescriptionV2 = [
    ('fFileSignature', 0, '4s'),
    ('fFileVersionNumber', 4, '4b'),
    ('uFileInfoSize', 8, 'I'),
    ('lActualEpisodes', 12, 'I'),
    ('uFileStartDate', 16, 'I'),
    ('uFileStartTimeMS', 20, 'I'),
    ('uStopwatchTime', 24, 'I'),
    ('nFileType', 28, 'H'),
    ('nDataFormat', 30, 'H'),
    ('nSimultaneousScan', 32, 'H'),
    ('nCRCEnable', 34, 'H'),
    ('uFileCRC', 36, 'I'),
    ('FileGUID', 40, 'I'),
    ('uCreatorVersion', 56, 'I'),
    ('uCreatorNameIndex', 60, 'I'),
    ('uModifierVersion', 64, 'I'),
    ('uModifierNameIndex', 68, 'I'),
    ('uProtocolPathIndex', 72, 'I'),
    ]


sectionNames = [
    'ProtocolSection',
    'ADCSection',
    'DACSection',
    'EpochSection',
    'ADCPerDACSection',
    'EpochPerDACSection',
    'UserListSection',
    'StatsRegionSection',
    'MathSection',
    'StringsSection',
    'DataSection',
    'TagSection',
    'ScopeSection',
    'DeltaSection',
    'VoiceTagSection',
    'SynchArraySection',
    'AnnotationSection',
    'StatsSection',
    ]


protocolInfoDescription = [
    ('nOperationMode', 'h'),
    ('fADCSequenceInterval', 'f'),
    ('bEnableFileCompression', 'b'),
    ('sUnused1', '3s'),
    ('uFileCompressionRatio', 'I'),
    ('fSynchTimeUnit', 'f'),
    ('fSecondsPerRun', 'f'),
    ('lNumSamplesPerEpisode', 'i'),
    ('lPreTriggerSamples', 'i'),
    ('lEpisodesPerRun', 'i'),
    ('lRunsPerTrial', 'i'),
    ('lNumberOfTrials', 'i'),
    ('nAveragingMode', 'h'),
    ('nUndoRunCount', 'h'),
    ('nFirstEpisodeInRun', 'h'),
    ('fTriggerThreshold', 'f'),
    ('nTriggerSource', 'h'),
    ('nTriggerAction', 'h'),
    ('nTriggerPolarity', 'h'),
    ('fScopeOutputInterval', 'f'),
    ('fEpisodeStartToStart', 'f'),
    ('fRunStartToStart', 'f'),
    ('lAverageCount', 'i'),
    ('fTrialStartToStart', 'f'),
    ('nAutoTriggerStrategy', 'h'),
    ('fFirstRunDelayS', 'f'),
    ('nChannelStatsStrategy', 'h'),
    ('lSamplesPerTrace', 'i'),
    ('lStartDisplayNum', 'i'),
    ('lFinishDisplayNum', 'i'),
    ('nShowPNRawData', 'h'),
    ('fStatisticsPeriod', 'f'),
    ('lStatisticsMeasurements', 'i'),
    ('nStatisticsSaveStrategy', 'h'),
    ('fADCRange', 'f'),
    ('fDACRange', 'f'),
    ('lADCResolution', 'i'),
    ('lDACResolution', 'i'),
    ('nExperimentType', 'h'),
    ('nManualInfoStrategy', 'h'),
    ('nCommentsEnable', 'h'),
    ('lFileCommentIndex', 'i'),
    ('nAutoAnalyseEnable', 'h'),
    ('nSignalType', 'h'),
    ('nDigitalEnable', 'h'),
    ('nActiveDACChannel', 'h'),
    ('nDigitalHolding', 'h'),
    ('nDigitalInterEpisode', 'h'),
    ('nDigitalDACChannel', 'h'),
    ('nDigitalTrainActiveLogic', 'h'),
    ('nStatsEnable', 'h'),
    ('nStatisticsClearStrategy', 'h'),
    ('nLevelHysteresis', 'h'),
    ('lTimeHysteresis', 'i'),
    ('nAllowExternalTags', 'h'),
    ('nAverageAlgorithm', 'h'),
    ('fAverageWeighting', 'f'),
    ('nUndoPromptStrategy', 'h'),
    ('nTrialTriggerSource', 'h'),
    ('nStatisticsDisplayStrategy', 'h'),
    ('nExternalTagType', 'h'),
    ('nScopeTriggerOut', 'h'),
    ('nLTPType', 'h'),
    ('nAlternateDACOutputState', 'h'),
    ('nAlternateDigitalOutputState', 'h'),
    ('fCellID', '3f'),
    ('nDigitizerADCs', 'h'),
    ('nDigitizerDACs', 'h'),
    ('nDigitizerTotalDigitalOuts', 'h'),
    ('nDigitizerSynchDigitalOuts', 'h'),
    ('nDigitizerType', 'h'),
    ]


ADCInfoDescription = [
    ('nADCNum', 'h'),
    ('nTelegraphEnable', 'h'),
    ('nTelegraphInstrument', 'h'),
    ('fTelegraphAdditGain', 'f'),
    ('fTelegraphFilter', 'f'),
    ('fTelegraphMembraneCap', 'f'),
    ('nTelegraphMode', 'h'),
    ('fTelegraphAccessResistance', 'f'),
    ('nADCPtoLChannelMap', 'h'),
    ('nADCSamplingSeq', 'h'),
    ('fADCProgrammableGain', 'f'),
    ('fADCDisplayAmplification', 'f'),
    ('fADCDisplayOffset', 'f'),
    ('fInstrumentScaleFactor', 'f'),
    ('fInstrumentOffset', 'f'),
    ('fSignalGain', 'f'),
    ('fSignalOffset', 'f'),
    ('fSignalLowpassFilter', 'f'),
    ('fSignalHighpassFilter', 'f'),
    ('nLowpassFilterType', 'b'),
    ('nHighpassFilterType', 'b'),
    ('fPostProcessLowpassFilter', 'f'),
    ('nPostProcessLowpassFilterType', 'c'),
    ('bEnabledDuringPN', 'b'),
    ('nStatsChannelPolarity', 'h'),
    ('lADCChannelNameIndex', 'i'),
    ('lADCUnitsIndex', 'i'),
    ]

TagInfoDescription = [
    ('lTagTime', 'i'),
    ('sComment', '56s'),
    ('nTagType', 'h'),
    ('nVoiceTagNumber_or_AnnotationIndex', 'h'),
    ]

DACInfoDescription = [
    ('nDACNum', 'h'),
    ('nTelegraphDACScaleFactorEnable', 'h'),
    ('fInstrumentHoldingLevel', 'f'),
    ('fDACScaleFactor', 'f'),
    ('fDACHoldingLevel', 'f'),
    ('fDACCalibrationFactor', 'f'),
    ('fDACCalibrationOffset', 'f'),
    ('lDACChannelNameIndex', 'i'),
    ('lDACChannelUnitsIndex', 'i'),
    ('lDACFilePtr', 'i'),
    ('lDACFileNumEpisodes', 'i'),
    ('nWaveformEnable', 'h'),
    ('nWaveformSource', 'h'),
    ('nInterEpisodeLevel', 'h'),
    ('fDACFileScale', 'f'),
    ('fDACFileOffset', 'f'),
    ('lDACFileEpisodeNum', 'i'),
    ('nDACFileADCNum', 'h'),
    ('nConditEnable', 'h'),
    ('lConditNumPulses', 'i'),
    ('fBaselineDuration', 'f'),
    ('fBaselineLevel', 'f'),
    ('fStepDuration', 'f'),
    ('fStepLevel', 'f'),
    ('fPostTrainPeriod', 'f'),
    ('fPostTrainLevel', 'f'),
    ('nMembTestEnable', 'h'),
    ('nLeakSubtractType', 'h'),
    ('nPNPolarity', 'h'),
    ('fPNHoldingLevel', 'f'),
    ('nPNNumADCChannels', 'h'),
    ('nPNPosition', 'h'),
    ('nPNNumPulses', 'h'),
    ('fPNSettlingTime', 'f'),
    ('fPNInterpulse', 'f'),
    ('nLTPUsageOfDAC', 'h'),
    ('nLTPPresynapticPulses', 'h'),
    ('lDACFilePathIndex', 'i'),
    ('fMembTestPreSettlingTimeMS', 'f'),
    ('fMembTestPostSettlingTimeMS', 'f'),
    ('nLeakSubtractADCIndex', 'h'),
    ('sUnused', '124s'),
    ]

EpochInfoPerDACDescription = [
    ('nEpochNum', 'h'),
    ('nDACNum', 'h'),
    ('nEpochType', 'h'),
    ('fEpochInitLevel', 'f'),
    ('fEpochLevelInc', 'f'),
    ('lEpochInitDuration', 'i'),
    ('lEpochDurationInc', 'i'),
    ('lEpochPulsePeriod', 'i'),
    ('lEpochPulseWidth', 'i'),
    ('sUnused', '18s'),
    ]

EpochInfoDescription = [
    ('nEpochNum', 'h'),
    ('nDigitalValue', 'h'),
    ('nDigitalTrainValue', 'h'),
    ('nAlternateDigitalValue', 'h'),
    ('nAlternateDigitalTrainValue', 'h'),
    ('bEpochCompression', 'b'),
    ('sUnused', '21s'),
    ]
