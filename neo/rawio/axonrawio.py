# -*- coding: utf-8 -*-
"""
Class for reading data from pCLAMP and AxoScope
files (.abf version 1 and 2), developed by Molecular device/Axon technologies.

- abf = Axon binary file
- atf is a text file based format from axon that could be
  read by AsciiIO (but this file is less efficient.)


This code is a port of abfload and abf2load
written in Matlab (BSD-2-Clause licence) by :
 - Copyright (c) 2009, Forrest Collman, fcollman@princeton.edu
 - Copyright (c) 2004, Harald Hentschke
and available here:
http://www.mathworks.com/matlabcentral/fileexchange/22114-abf2load

Information on abf 1 and 2 formats is available here:
http://www.moleculardevices.com/pages/software/developer_info.html

This file supports the old (ABF1) and new (ABF2) format.
ABF1 (clampfit <=9) and ABF2 (clampfit >10)

All possible mode are possible :
    - event-driven variable-length mode 1 -> return several Segments per Block
    - event-driven fixed-length mode 2 or 5 -> return several Segments
    - gap free mode -> return one (or sevral) Segment in the Block

Supported : Read

Author: Samuel Garcia, JS Nowacki

Note: j.s.nowacki@gmail.com has a C++ library with SWIG bindings which also
reads abf files - would be good to cross-check

"""
from __future__ import  print_function, division, absolute_import
#from __future__ import unicode_literals is not compatible with numpy.dtype both py2 py3

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype, 
        _event_channel_dtype)

import numpy as np

import struct
import datetime
import os
from io import open, BufferedReader

import numpy as np



class AxonRawIO(BaseRawIO):
    
    extensions = ['abf']
    rawmode = 'one-file'
    
    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename
    
    def _parse_header(self):
        info = self._axon_info = parse_axon_soup(self.filename)
        
        version = info['fFileVersionNumber']

        # file format
        if info['nDataFormat'] == 0:
            sig_dtype = np.dtype('i2')
        elif info['nDataFormat'] == 1:
            sig_dtype = np.dtype('f4')

        if version < 2.:
            nbchannel = info['nADCNumChannels']
            head_offset = info['lDataSectionPtr'] * BLOCKSIZE + info[
                'nNumPointsIgnored'] * sig_dtype.itemsize
            totalsize = info['lActualAcqLength']
        elif version >= 2.:
            nbchannel = info['sections']['ADCSection']['llNumEntries']
            head_offset = info['sections']['DataSection'][
                'uBlockIndex'] * BLOCKSIZE
            totalsize = info['sections']['DataSection']['llNumEntries']

        self._raw_data = np.memmap(self.filename, dtype=sig_dtype, mode='r',
                        shape=(totalsize,), offset=head_offset)
        
        # 3 possible modes
        if version < 2.:
            mode = info['nOperationMode']
        elif version >= 2.:
            mode = info['protocol']['nOperationMode']
        
        assert mode in [1, 2, 3, 5], 'Mode {} is not supported'.formagt(mode)
        # event-driven variable-length mode (mode 1)
        # event-driven fixed-length mode (mode 2 or 5)
        # gap free mode (mode 3) can be in several episodes

        # read sweep pos
        if version < 2.:
            nbepisod = info['lSynchArraySize']
            offset_episode = info['lSynchArrayPtr'] * BLOCKSIZE
        elif version >= 2.:
            nbepisod = info['sections']['SynchArraySection'][
                'llNumEntries']
            offset_episode = info['sections']['SynchArraySection'][
                'uBlockIndex'] * BLOCKSIZE
        if nbepisod > 0:
            episode_array = np.memmap(
                self.filename, [('offset', 'i4'), ('len', 'i4')], 'r',
                shape=nbepisod, offset=offset_episode)
        else:
            episode_array = np.empty(1, [('offset', 'i4'), ('len', 'i4')])
            episode_array[0]['len'] = self._raw_data.size
            episode_array[0]['offset'] = 0

        # sampling_rate
        if version < 2.:
            self._sampling_rate = 1. / (info['fADCSampleInterval'] *
                                  nbchannel * 1.e-6)
        elif version >= 2.:
            self._sampling_rate = 1.e6 / \
                info['protocol']['fADCSequenceInterval']
        
        # one sweep = one segment
        nb_segment = episode_array.size
        
        #Get raw data by segment
        self._raw_signals = {}
        self._t_starts = {}
        pos = 0
        for seg_index in range(nb_segment):
            length = episode_array[seg_index]['len']

            if version < 2.:
                fSynchTimeUnit = info['fSynchTimeUnit']
            elif version >= 2.:
                fSynchTimeUnit = info['protocol']['fSynchTimeUnit']

            if (fSynchTimeUnit != 0) and (mode == 1):
                length /= fSynchTimeUnit

            self._raw_signals[seg_index] = self._raw_data[pos:pos+length].reshape(-1, nbchannel)
            pos += length
            
            t_start = float(episode_array[seg_index]['offset'])
            if (fSynchTimeUnit == 0):
                    t_start =  t_start / self._sampling_rate
            else:
                    t_start = t_start * fSynchTimeUnit *1e-6
            self._t_starts[seg_index] = t_start
            
            
        #Create channel header
        if version < 2.:
            channel_ids = [chan_num for chan_num in
                     info['nADCSamplingSeq'] if chan_num >= 0]
        else:
            channel_ids = list(range(nbchannel))
        
        sig_channels =[]
        adc_nums = []
        for chan_index, chan_id in enumerate(channel_ids):
            if version < 2.:
                name = info['sADCChannelName'][chan_id].replace(b' ', b'')
                units = info['sADCUnits'][chan_id].replace(b'\xb5', b'u').\
                    replace(b' ', b'').decode('utf-8')  # \xb5 is µ
                adc_num = info['nADCPtoLChannelMap'][chan_id]
            elif version >= 2.:
                ADCInfo = info['listADCInfo'][chan_id]
                name = ADCInfo['ADCChNames'].replace(b' ', b'')
                units = ADCInfo['ADCChUnits'].replace(b'\xb5', b'u').\
                    replace(b' ', b'').decode('utf-8')
                adc_num = ADCInfo['nADCNum']
            adc_nums.append(adc_num)
            
            if version<2.:
                gain = info['fADCRange']
                gain /= info['fInstrumentScaleFactor'][chan_id]
                gain /= info['fSignalGain'][chan_id]
                gain /= info['fADCProgrammableGain'][chan_id]
                gain /= info['lADCResolution']
                if info['nTelegraphEnable'][chan_id]:
                    gain /=info['fTelegraphAdditGain'][chan_id]
                offset = info['fInstrumentOffset'][chan_id] 
                offset -= info['fSignalOffset'][chan_id]
            elif version>=2.:
                gain = info['protocol']['fADCRange']
                gain /= info['listADCInfo'][chan_id]['fInstrumentScaleFactor']
                gain /= info['listADCInfo'][chan_id]['fSignalGain']
                gain /= info['listADCInfo'][chan_id]['fADCProgrammableGain']
                gain /= info['protocol']['lADCResolution']
                if info['listADCInfo'][chan_id]['nTelegraphEnable']:
                    gain /= info['listADCInfo'][chan_id]['fTelegraphAdditGain']
                offset = info['listADCInfo'][chan_id]['fInstrumentOffset']
                offset -= info['listADCInfo'][chan_id]['fSignalOffset']
            group_id = 0
            sig_channels.append((name, chan_id, self._sampling_rate, sig_dtype, units, offset, gain, group_id))
            
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        
        # only one events channel : tag
        if mode in [3, 5]:  # TODO check if tags exits in other mode
            #In ABF timstamps are not attached too any particular segment
            # so each segment acess all event
            timestamps = []
            labels = []
            comments = []
            for i, tag in enumerate(info['listTag']):
                timestamps.append(tag['lTagTime'])
                labels.append(str(tag['nTagType']))
                comments.append(clean_string(tag['sComment']))
            self._raw_ev_timestamps = np.array(timestamps)
            self._ev_labels = np.array(labels, dtype='U')
            self._ev_comments = np.array(comments, dtype='U')
        
        event_channels = []
        event_channels.append(('Tag', '', 'event'))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        
        #No spikes
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        
        #fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [nb_segment]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels
        
        # insert some annotation at some place
        self._generate_minimal_annotations()
        bl_annotations = self.raw_annotations['blocks'][0]
        
        
        bl_annotations['rec_datetime'] = info['rec_datetime']
        bl_annotations['abf_version'] = version
        
        for seg_index in range(nb_segment):
            seg_annotations = bl_annotations['segments'][seg_index]
            seg_annotations['abf_version'] = version
        
            for c in range(sig_channels.size):
                anasig_an = seg_annotations['signals'][c]
                anasig_an['nADCNum'] = adc_nums[c]

            for c in range(event_channels.size):
                ev_ann = seg_annotations['events'][c]
                ev_ann['comments'] = self._ev_comments
        
    def _source_name(self):
        return self.filename
    
    def _segment_t_start(self, block_index, seg_index):
        return self._t_starts[seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._t_starts[seg_index] + self._raw_signals[seg_index].shape[0]/self._sampling_rate
        return t_stop
    
    def get_signal_size(self, block_index, seg_index, channel_indexes):
        shape = self._raw_signals[seg_index].shape
        return shape[0]
    
    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return self._t_starts[seg_index]
    
    def _get_analogsignal_chunk(self, block_index, seg_index,  i_start, i_stop, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_signals[seg_index][slice(i_start, i_stop), channel_indexes]
        return raw_signals
    
    def _event_count(self, block_index, seg_index, event_channel_index):
        return self._raw_ev_timestamps.size
    
    def _get_event_timestamps(self,  block_index, seg_index, event_channel_index, t_start, t_stop):
        #In ABF timstamps are not attached too any particular segment
        # so each segmetn acees all event
        timestamp = self._raw_ev_timestamps
        labels = self._ev_labels
        durations = None
        
        if t_start is not None:
            keep = timestamp>=int(t_start*self._sampling_rate)
            timestamp = timestamp[keep]
            labels = labels[keep]
        
        if t_stop is not None:
            keep = timestamp<=int(t_stop*self._sampling_rate)
            timestamp = timestamp[keep]
            labels = labels[keep]
        
        return timestamp, durations, labels        
        
    def _rescale_event_timestamp(self, event_timestamps, dtype):
        event_times = event_timestamps.astype(dtype)/self._sampling_rate
        return event_times

    def read_raw_protocol(self):
        """
        Read the protocol waveform of the file, if present;
        function works with ABF2 only. Protocols can be reconstructed
        from the ABF1 header.

        Returns: list of segments (one for every episode)
                 with list of analog signls (one for every DAC).
        
        Author:  JS Nowacki
        """
        info = self._axon_info

        if info['fFileVersionNumber'] < 2.:
            raise IOError("Protocol section is only present in ABF2 files.")

        nADC = info['sections']['ADCSection'][
            'llNumEntries']  # Number of ADC channels
        nDAC = info['sections']['DACSection'][
            'llNumEntries']  # Number of DAC channels
        nSam = int(info['protocol'][
            'lNumSamplesPerEpisode'] / nADC)  # Number of samples per episode
        nEpi = info['lActualEpisodes']  # Actual number of episodes
        
        #~ print('read_raw_protocol', nEpi, nDAC)

        # Make a list of segments with analog signals with just holding levels
        # List of segments relates to number of episodes, as for recorded data
        sigs_by_segments = []
        for epiNum in range(nEpi):
            # One analog signal for each DAC in segment (episode)
            signals = []
            for DACNum in range(nDAC):
                sig = np.ones(nSam) * info['listDACInfo'][DACNum]['fDACHoldingLevel']
                # If there are epoch infos for this DAC
                if DACNum in info['dictEpochInfoPerDAC']:
                    # Save last sample index
                    i_last = int(nSam * 15625 / 10**6)
                    # TODO guess for first holding
                    # Go over EpochInfoPerDAC and change the analog signal
                    # according to the epochs
                    epochInfo = info['dictEpochInfoPerDAC'][DACNum]
                    for epochNum, epoch in epochInfo.items():
                        i_begin = i_last
                        i_end = i_last + epoch['lEpochInitDuration'] +\
                            epoch['lEpochDurationInc'] * epiNum
                        dif = i_end-i_begin
                        sig[i_begin:i_end] = np.ones((dif)) *\
                            (epoch['fEpochInitLevel'] +epoch['fEpochLevelInc'] *
                                                    epiNum)
                        i_last += epoch['lEpochInitDuration'] +\
                            epoch['lEpochDurationInc'] * epiNum
                signals.append(sig)
            sigs_by_segments.append(signals)
        
        sig_names = []
        sig_units = []
        for DACNum in range(nDAC):
            name = info['listDACInfo'][DACNum]['DACChNames'].decode("utf-8")
            units = info['listDACInfo'][DACNum]['DACChUnits'].\
            replace(b'\xb5', b'u').decode('utf-8')  # \xb5 is µ
            sig_names.append(name)
            sig_units.append(units)

        return sigs_by_segments, sig_names, sig_units




def parse_axon_soup(filename):
    """
    read the header of the file

    The strategy here differs from the original script under Matlab.
    In the original script for ABF2, it completes the header with
    information that is located in other structures.

    In ABF2 this function returns info with sub dict:
        sections             (ABF2)
        protocol             (ABF2)
        listTags             (ABF1&2)
        listADCInfo          (ABF2)
        listDACInfo          (ABF2)
        dictEpochInfoPerDAC  (ABF2)
    that contains more information.
    """
    with open(filename, 'rb') as fid:
        f = StructFile(fid)

    
        # version
        f_file_signature = f.read(4)
        if f_file_signature == b'ABF ':
            header_description = headerDescriptionV1
        elif f_file_signature == b'ABF2':
            header_description = headerDescriptionV2
        else:
            return None

        # construct dict
        header = {}
        for key, offset, fmt in header_description:
            val = f.read_f(fmt, offset=offset)
            if len(val) == 1:
                header[key] = val[0]
            else:
                header[key] = np.array(val)

        # correction of version number and starttime
        if f_file_signature == b'ABF ':
            header['lFileStartTime'] += header[
                'nFileStartMillisecs'] * .001
        elif f_file_signature == b'ABF2':
            n = header['fFileVersionNumber']
            header['fFileVersionNumber'] = n[3] + 0.1 * n[2] +\
                0.01 * n[1] + 0.001 * n[0]
            header['lFileStartTime'] = header['uFileStartTimeMS'] * .001

        if header['fFileVersionNumber'] < 2.:
            # tags
            listTag = []
            for i in range(header['lNumTagEntries']):
                f.seek(header['lTagSectionPtr'] + i * 64)
                tag = {}
                for key, fmt in TagInfoDescription:
                    val = f.read_f(fmt)
                    if len(val) == 1:
                        tag[key] = val[0]
                    else:
                        tag[key] = np.array(val)
                listTag.append(tag)
            header['listTag'] = listTag
            #protocol name formatting
            header['sProtocolPath'] = clean_string(header['sProtocolPath'])
            header['sProtocolPath'] = header['sProtocolPath'].\
                replace(b'\\', b'/')

        elif header['fFileVersionNumber'] >= 2.:
            # in abf2 some info are in other place

            # sections
            sections = {}
            for s, sectionName in enumerate(sectionNames):
                uBlockIndex, uBytes, llNumEntries =\
                    f.read_f('IIl', offset=76 + s * 16)
                sections[sectionName] = {}
                sections[sectionName]['uBlockIndex'] = uBlockIndex
                sections[sectionName]['uBytes'] = uBytes
                sections[sectionName]['llNumEntries'] = llNumEntries
            header['sections'] = sections

            # strings sections
            # hack for reading channels names and units
            f.seek(sections['StringsSection']['uBlockIndex'] * BLOCKSIZE)
            big_string = f.read(sections['StringsSection']['uBytes'])
            goodstart=-1
            for key in [b'AXENGN', b'clampex', b'Clampex', b'CLAMPEX', b'axoscope']:
                #goodstart = big_string.lower().find(key)
                goodstart = big_string.find(key)
                if goodstart!=-1: break
            assert goodstart!=-1, 'This file does not contain clampex, axoscope or clampfit in the header'
            big_string = big_string[goodstart:]
            strings = big_string.split(b'\x00')

            # ADC sections
            header['listADCInfo'] = []
            for i in range(sections['ADCSection']['llNumEntries']):
                # read ADCInfo
                f.seek(sections['ADCSection']['uBlockIndex'] *
                         BLOCKSIZE + sections['ADCSection']['uBytes'] * i)
                ADCInfo = {}
                for key, fmt in ADCInfoDescription:
                    val = f.read_f(fmt)
                    if len(val) == 1:
                        ADCInfo[key] = val[0]
                    else:
                        ADCInfo[key] = np.array(val)
                ADCInfo['ADCChNames'] = strings[ADCInfo['lADCChannelNameIndex'] - 1]
                ADCInfo['ADCChUnits'] = strings[ADCInfo['lADCUnitsIndex'] - 1]
                header['listADCInfo'].append(ADCInfo)

            # protocol sections
            protocol = {}
            f.seek(sections['ProtocolSection']['uBlockIndex'] * BLOCKSIZE)
            for key, fmt in protocolInfoDescription:
                val = f.read_f(fmt)
                if len(val) == 1:
                    protocol[key] = val[0]
                else:
                    protocol[key] = np.array(val)
            header['protocol'] = protocol
            header['sProtocolPath'] = strings[header['uProtocolPathIndex']-1]

            # tags
            listTag = []
            for i in range(sections['TagSection']['llNumEntries']):
                f.seek(sections['TagSection']['uBlockIndex'] *
                         BLOCKSIZE + sections['TagSection']['uBytes'] * i)
                tag = {}
                for key, fmt in TagInfoDescription:
                    val = f.read_f(fmt)
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
                f.seek(sections['DACSection']['uBlockIndex'] *
                         BLOCKSIZE + sections['DACSection']['uBytes'] * i)
                DACInfo = {}
                for key, fmt in DACInfoDescription:
                    val = f.read_f(fmt)
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
                f.seek(sections['EpochPerDACSection']['uBlockIndex'] *
                         BLOCKSIZE +
                         sections['EpochPerDACSection']['uBytes'] * i)
                EpochInfoPerDAC = {}
                for key, fmt in EpochInfoPerDACDescription:
                    val = f.read_f(fmt)
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

        # date and time
        if header['fFileVersionNumber'] < 2.:
            YY = 1900
            MM = 1
            DD = 1
            hh = int(header['lFileStartTime'] / 3600.)
            mm = int((header['lFileStartTime'] - hh * 3600) / 60)
            ss = header['lFileStartTime'] - hh * 3600 - mm * 60
            ms = int(np.mod(ss, 1) * 1e6)
            ss = int(ss)
        elif header['fFileVersionNumber'] >= 2.:
            YY = int(header['uFileStartDate'] / 10000)
            MM = int((header['uFileStartDate'] - YY * 10000) / 100)
            DD = int(header['uFileStartDate'] - YY * 10000 - MM * 100)
            hh = int(header['uFileStartTimeMS'] / 1000. / 3600.)
            mm = int((header['uFileStartTimeMS'] / 1000. - hh * 3600) / 60)
            ss = header['uFileStartTimeMS'] / 1000. - hh * 3600 - mm * 60
            ms = int(np.mod(ss, 1) * 1e6)
            ss = int(ss)
        header['rec_datetime'] = datetime.datetime(YY, MM, DD, hh, mm, ss, ms)


    return header    


class StructFile(BufferedReader):
    def read_f(self, fmt, offset=None):
        if offset is not None:
            self.seek(offset)
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

def clean_string(s):
    s = s.rstrip(b'\x00')
    s = s.rstrip(b' ')
    return s

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