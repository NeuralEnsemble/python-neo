# pypl2lib.py - Classes and functions for accessing functions
# in PL2FileReader.dll
#
# (c) 2016 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

# This version of the pypl2lib was modified from https://gitlab.com/penalab/pypl2
# Thanks to Roland "Elpy" Ferger for implementing the original updates.

from sys import platform
if any(platform.startswith(name) for name in ('linux', 'darwin', 'freebsd')):
        import zugbruecke.ctypes as ctypes
elif platform.startswith('win'):
        import ctypes
else:
        raise SystemError('unsupported platform')

from ctypes import (c_int, c_char, c_double, c_uint, c_ushort, c_ulonglong,
                    byref)
import os
import platform


class tm(ctypes.Structure):
    _fields_ = [("tm_sec", c_int),
                ("tm_min", c_int),
                ("tm_hour", c_int),
                ("tm_mday", c_int),
                ("tm_mon", c_int),
                ("tm_year", c_int),
                ("tm_wday", c_int),
                ("tm_yday", c_int),
                ("tm_isdst", c_int)]


class PL2FileInfo(ctypes.Structure):
    _fields_ = [("m_CreatorComment", c_char * 256),
                ("m_CreatorSoftwareName", c_char * 64),
                ("m_CreatorSoftwareVersion", c_char * 16),
                ("m_CreatorDateTime", tm),
                ("m_CreatorDateTimeMilliseconds", c_int),
                ("m_TimestampFrequency", c_double),
                ("m_NumberOfChannelHeaders", c_uint),
                ("m_TotalNumberOfSpikeChannels", c_uint),
                ("m_NumberOfRecordedSpikeChannels", c_uint),
                ("m_TotalNumberOfAnalogChannels", c_uint),
                ("m_NumberOFRecordedAnalogChannels", c_uint),
                ("m_NumberOfDigitalChannels", c_uint),
                ("m_MinimumTrodality", c_uint),
                ("m_MaximumTrodality", c_uint),
                ("m_NumberOfNonOmniPlexSources", c_uint),
                ("m_Unused", c_int),
                ("m_ReprocessorComment", c_char * 256),
                ("m_ReprocessorSoftwareName", c_char * 64),
                ("m_ReprocessorSoftwareVersion", c_char * 16),
                ("m_ReprocessorDateTime", tm),
                ("m_ReprocessorDateTimeMilliseconds", c_int),
                ("m_StartRecordingTime", c_ulonglong),
                ("m_DurationOfRecording", c_ulonglong)]


class PL2AnalogChannelInfo(ctypes.Structure):
    _fields_ = [("m_Name", c_char * 64),
                ("m_Source", c_uint),
                ("m_Channel", c_uint),
                ("m_ChannelEnabled", c_uint),
                ("m_ChannelRecordingEnabled", c_uint),
                ("m_Units", c_char * 16),
                ("m_SamplesPerSecond", c_double),
                ("m_CoeffToConvertToUnits", c_double),
                ("m_SourceTrodality", c_uint),
                ("m_OneBasedTrode", c_ushort),
                ("m_OneBasedChannelInTrode", c_ushort),
                ("m_NumberOfValues", c_ulonglong),
                ("m_MaximumNumberOfFragments", c_ulonglong)]


class PL2SpikeChannelInfo(ctypes.Structure):
    _fields_ = [("m_Name", c_char * 64),
                ("m_Source", c_uint),
                ("m_Channel", c_uint),
                ("m_ChannelEnabled", c_uint),
                ("m_ChannelRecordingEnabled", c_uint),
                ("m_Units", c_char * 16),
                ("m_SamplesPerSecond", c_double),
                ("m_CoeffToConvertToUnits", c_double),
                ("m_SamplesPerSpike", c_uint),
                ("m_Threshold", c_int),
                ("m_PreThresholdSamples", c_uint),
                ("m_SortEnabled", c_uint),
                ("m_SortMethod", c_uint),
                ("m_NumberOfUnits", c_uint),
                ("m_SortRangeStart", c_uint),
                ("m_SortRangeEnd", c_uint),
                ("m_UnitCounts", c_ulonglong * 256),
                ("m_SourceTrodality", c_uint),
                ("m_OneBasedTrode", c_ushort),
                ("m_OneBasedChannelInTrode", c_ushort),
                ("m_NumberOfSpikes", c_ulonglong)]


class PL2DigitalChannelInfo(ctypes.Structure):
    _fields_ = [("m_Name", c_char * 64),
                ("m_Source", c_uint),
                ("m_Channel", c_uint),
                ("m_ChannelEnabled", c_uint),
                ("m_ChannelRecordingEnabled", c_uint),
                ("m_NumberOfEvents", c_ulonglong)]


class PyPL2FileReader:
    def __init__(self, pl2_dll_path=None):
        """
        PyPL2FileReader class implements functions in the C++ PL2 File Reader
        API provided by Plexon, Inc.
        
        Args:
            pl2_dll_path - path where PL2FileReader.dll is location.
                The default value assumes the .dll files are located in the
                'bin' directory, which is a subdirectory of the directory this
                script is in.
                Any file path passed is converted to an absolute path and checked
                to see if the .dll exists there.
        
        Returns:
            None
        """
        self.platform = platform.architecture()[0]
        if pl2_dll_path is None:
            pl2_dll_path = os.path.join(os.path.split(__file__)[0], 'bin')
        self.pl2_dll_path = os.path.abspath(pl2_dll_path)

        # if self.platform == '32bit':
        self.pl2_dll_file = os.path.join(self.pl2_dll_path, 'PL2FileReader.dll')
        # else:
        #     self.pl2_dll_file = os.path.join(self.pl2_dll_path, 'PL2FileReader64.dll')

        try:
            self.pl2_dll = ctypes.CDLL(self.pl2_dll_file)
        except IOError:
            raise IOError("Error: Can't load PL2FileReader.dll at: " + self.pl2_dll_file,
            "PL2FileReader.dll is bundled with the C++ PL2 Offline Files SDK",
            "located on the Plexon Inc website: www.plexon.com",
            "Contact Plexon Support for more information: support@plexon.com")

    def pl2_open_file(self, pl2_file):
        """
        Opens and returns a handle to a PL2 file.
        
        Args:
            pl2_file - full path of the file
            file_handle - file handle
            
        Returns:
            file_handle > 0 if success
            file_handle = 0 if failure
            
        """
        self.file_handle = c_int(0)
        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_OpenFile(pl2_file.encode('ascii'),
                                                byref(self.file_handle))

        return self.file_handle.value

    def pl2_close_file(self, file_handle):
        """
        Closes handle to PL2 file.
        
        Args:
            file_handle - file handle of file to be closed
            
        Returns:
            None
        """

        self.pl2_dll.PL2_CloseFile(self.file_handle)

    def pl2_close_all_files(self):
        """
        Closes all files that have been opened by the .dll
        
        Args:
            None
        
        Returns:
            None
        """

        self.pl2_dll.PL2_CloseAllFiles()

    def pl2_get_last_error(self, buffer, buffer_size):
        """
        Retrieve description of the last error
        
        Args:
            buffer - instance of c_char array
            buffer_size - size of buffer
        
        Returns:
            1 - Success
            0 - Failure
            buffer is filled with error message
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetLastError(byref(buffer), c_int(buffer_size))

        return self.result

    def pl2_get_file_info(self, file_handle, pl2_file_info):
        """
        Retrieve information about pl2 file.
        
        Args:
            file_handle - file handle
            pl2_file_info - PL2FileInfo class instance
        
        Returns:
            1 - Success
            0 - Failure
            The instance of PL2FileInfo passed to function is filled with file info
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetFileInfo(c_int(file_handle), byref(pl2_file_info))

        return self.result

    def pl2_get_analog_channel_info(self, file_handle, zero_based_channel_index,
                                    pl2_analog_channel_info):
        """
        Retrieve information about an analog channel
        
        Args:
            file_handle - file handle
            zero_based_channel_index - zero-based analog channel index
            pl2_analog_channel_info - PL2AnalogChannelInfo class instance
        
        Returns:
            1 - Success
            0 - Failure
            The instance of PL2AnalogChannelInfo passed to function is filled with channel info
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetAnalogChannelInfo(c_int(file_handle),
                                                            zero_based_channel_index,
                                                            byref(pl2_analog_channel_info))

        return self.result

    def pl2_get_analog_channel_info_by_name(self, file_handle, channel_name,
                                            pl2_analog_channel_info):
        """
        Retrieve information about an analog channel
        
        Args:
            file_handle - file handle
            channel_name - analog channel name
            pl2_analog_channel_info - PL2AnalogChannelInfo class instance
        
        Returns:
            1 - Success
            0 - Failure        
            The instance of PL2AnalogChannelInfo is filled with channel info
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetAnalogChannelInfoByName(c_int(file_handle),
                                                                  channel_name.encode('ascii'),
                                                                  byref(pl2_analog_channel_info))

        return self.result

    def pl2_get_analog_channel_info_by_source(self, file_handle, source_id,
                                              one_based_channel_index_in_source,
                                              pl2_analog_channel_info):
        """
        Retrieve information about an analog channel
        
        Args:
            file_handle - file handle
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source
            pl2_analog_channel_info - PL2AnalogChannelInfo class instance
        
        Returns:
            1 - Success
            0 - Failure        
            The instance of PL2AnalogChannelInfo is filled with channel info
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetAnalogChannelInfoBySource(c_int(file_handle),
                                                                    c_int(source_id), c_int(
                one_based_channel_index_in_source), byref(pl2_analog_channel_info))

        return self.result

    def pl2_get_analog_channel_data(self, file_handle, zero_based_channel_index,
                                    num_fragments_returned, num_data_points_returned,
                                    fragment_timestamps, fragment_counts, values):
        """
        Retrieve analog channel data
        
        Args:
            file_handle - file handle
            zero_based_channel_index - zero based channel index
            num_fragments_returned - c_ulonglong class instance
            num_data_points_returned - c_ulonglong class instance
            fragment_timestamps - c_longlong class instance array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            fragment_counts - c_ulonglong class instance array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            values - c_short class instance array the size of PL2AnalogChannelInfo.m_NumberOfValues
            
        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetAnalogChannelData(c_int(file_handle),
                                                            c_int(zero_based_channel_index),
                                                            byref(num_fragments_returned),
                                                            byref(num_data_points_returned),
                                                            byref(fragment_timestamps),
                                                            byref(fragment_counts), byref(values))

        return self.result

    def pl2_get_analog_channel_data_subset(self):
        pass

    def pl2_get_analog_channel_data_by_name(self, file_handle, channel_name,
                                            num_fragments_returned, num_data_points_returned,
                                            fragment_timestamps, fragment_counts, values):
        """
        Retrieve analog channel data
        
        Args:
            file_handle - file handle
            channel_name - analog channel name
            num_fragments_returned - c_ulonglong class instance
            num_data_points_returned - c_ulonglong class instance
            fragment_timestamps - c_longlong class instance array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            fragment_counts - c_ulonglong class instance array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            values - c_short class instance array the size of PL2AnalogChannelInfo.m_NumberOfValues
            
        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetAnalogChannelDataByName(c_int(file_handle),
                                                                  channel_name.encode('ascii'),
                                                                  byref(num_fragments_returned),
                                                                  byref(num_data_points_returned),
                                                                  byref(fragment_timestamps),
                                                                  byref(fragment_counts),
                                                                  byref(values))

        return self.result

    def pl2_get_analog_channel_data_by_source(self, file_handle, source_id,
                                              one_based_channel_index_in_source,
                                              num_fragments_returned, num_data_points_returned,
                                              fragment_timestamps, fragment_counts, values):
        """
        Retrieve analog channel data
        
        Args:
            file_handle - file handle
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source
            num_fragments_returned - c_ulonglong class instance
            num_data_points_returned - c_ulonglong class instance
            fragment_timestamps - c_longlong class instance array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            fragment_counts - c_ulonglong class instance array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            values - c_short class instance array the size of PL2AnalogChannelInfo.m_NumberOfValues
            
        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetAnalogChannelDataBySource(c_int(file_handle),
                                                                    c_int(source_id), c_int(
                one_based_channel_index_in_source), byref(num_fragments_returned), byref(
                num_data_points_returned), byref(fragment_timestamps), byref(fragment_counts),
                                                                    byref(values))

        return self.result

    def pl2_get_spike_channel_info(self, file_handle, zero_based_channel_index,
                                   pl2_spike_channel_info):
        """
        Retrieve information about a spike channel
        
        Args:
            file_handle - file handle
            zero_based_channel_index - zero-based spike channel index
            pl2_spike_channel_info - PL2SpikeChannelInfo class instance
        
        Returns:
            1 - Success
            0 - Failure
            The instance of PL2SpikeChannelInfo passed to function is filled with channel info
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetSpikeChannelInfo(c_int(file_handle),
                                                           c_int(zero_based_channel_index),
                                                           byref(pl2_spike_channel_info))

        return self.result

    def pl2_get_spike_channel_info_by_name(self, file_handle, channel_name,
                                           pl2_spike_channel_info):
        """
        Retrieve information about a spike channel
        
        Args:
            file_handle - file handle
            channel_name - spike channel name
            pl2_spike_channel_info - PL2SpikeChannelInfo class instance
        
        Returns:
            1 - Success
            0 - Failure
            The instance of PL2SpikeChannelInfo passed to function is filled with channel info
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetSpikeChannelInfoByName(c_int(file_handle),
                                                                 channel_name.encode('ascii'),
                                                                 byref(pl2_spike_channel_info))

        return self.result

    def pl2_get_spike_channel_info_by_source(self, file_handle, source_id,
                                             one_based_channel_index_in_source,
                                             pl2_spike_channel_info):
        """
        Retrieve information about a spike channel
        
        Args:
            file_handle - file handle
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source
            pl2_spike_channel_info - PL2SpikeChannelInfo class instance
        
        Returns:
            1 - Success
            0 - Failure
            The instance of PL2SpikeChannelInfo passed to function is filled with channel info
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetSpikeChannelInfoBySource(c_int(file_handle),
                                                                   c_int(source_id), c_int(
                one_based_channel_index_in_source), byref(pl2_spike_channel_info))

        return self.result

    def pl2_get_spike_channel_data(self, file_handle, zero_based_channel_index,
                                   num_spikes_returned, spike_timestamps, units, values):
        """
        Retrieve spike channel data
        
        Args:
            file_handle - file handle
            zero_based_channel_index - zero based channel index
            num_spikes_returned - c_ulonglong class instance
            spike_timestamps - c_ulonglong class instance array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            units - c_ushort class instance array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            values - c_short class instance array the size of (PL2SpikeChannelInfo.m_NumberOfSpikes * PL2SpikeChannelInfo.m_SamplesPerSpike)
        
        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetSpikeChannelData(c_int(file_handle),
                                                           c_int(zero_based_channel_index),
                                                           byref(num_spikes_returned),
                                                           byref(spike_timestamps), byref(units),
                                                           byref(values))

        return self.result

    def pl2_get_spike_channel_data_by_name(self, file_handle, channel_name, num_spikes_returned,
                                           spike_timestamps, units, values):
        """
        Retrieve spike channel data
        
        Args:
            file_handle - file handle
            channel_name = channel name
            num_spikes_returned - c_ulonglong class instance
            spike_timestamps - c_ulonglong class instance array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            units - c_ushort class instance array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            values - c_short class instance array the size of (PL2SpikeChannelInfo.m_NumberOfSpikes * PL2SpikeChannelInfo.m_SamplesPerSpike)
        
        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetSpikeChannelDataByName(c_int(file_handle),
                                                                 channel_name.encode('ascii'),
                                                                 byref(num_spikes_returned),
                                                                 byref(spike_timestamps),
                                                                 byref(units), byref(values))

        return self.result

    def pl2_get_spike_channel_data_by_source(self, file_handle, source_id,
                                             one_based_channel_index_in_source,
                                             num_spikes_returned, spike_timestamps, units, values):
        """
        Retrieve spike channel data
        
        Args:
            file_handle - file handle
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source
            num_spikes_returned - c_ulonglong class instance
            spike_timestamps - c_ulonglong class instance array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            units - c_ushort class instance array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            values - c_short class instance array the size of (PL2SpikeChannelInfo.m_NumberOfSpikes * PL2SpikeChannelInfo.m_SamplesPerSpike)
        
        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetSpikeChannelDataBySource(c_int(file_handle),
                                                                   c_int(source_id), c_int(
                one_based_channel_index_in_source), byref(num_spikes_returned),
                                                                   byref(spike_timestamps),
                                                                   byref(units), byref(values))

        return self.result

    def pl2_get_digital_channel_info(self, file_handle, zero_based_channel_index,
                                     pl2_digital_channel_info):
        """
        Retrieve information about a digital event channel
        
        Args:
            file_handle - file handle
            zero_based_channel_index - zero-based digital event channel index
            pl2_digital_channel_info - PL2DigitalChannelInfo class instance
        
        Returns:
            1 - Success
            0 - Failure
            The instance of PL2DigitalChannelInfo passed to function is filled with channel info
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetDigitalChannelInfo(c_int(file_handle),
                                                             c_int(zero_based_channel_index),
                                                             byref(pl2_digital_channel_info))

        return self.result

    def pl2_get_digital_channel_info_by_name(self, file_handle, channel_name,
                                             pl2_digital_channel_info):
        """
        Retrieve information about a digital event channel
        
        Args:
            file_handle - file handle
            channel_name - digital event channel name
            pl2_digital_channel_info - PL2DigitalChannelInfo class instance
        
        Returns:
            1 - Success
            0 - Failure
            The instance of PL2DigitalChannelInfo passed to function is filled with channel info
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetDigitalChannelInfoByName(c_int(file_handle),
                                                                   channel_name.encode('ascii'),
                                                                   byref(pl2_digital_channel_info))

        return self.result

    def pl2_get_digital_channel_info_by_source(self, file_handle, source_id,
                                               one_based_channel_index_in_source,
                                               pl2_digital_channel_info):
        """
        Retrieve information about a digital event channel
        
        Args:
            file_handle - file handle
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source
            pl2_digital_channel_info - PL2DigitalChannelInfo class instance
        
        Returns:
            1 - Success
            0 - Failure
            The instance of PL2DigitalChannelInfo passed to function is filled with channel info
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetDigitalChannelInfoBySource(c_int(file_handle),
                                                                     c_int(source_id), c_int(
                one_based_channel_index_in_source), byref(pl2_digital_channel_info))

        return self.result

    def pl2_get_digital_channel_data(self, file_handle, zero_based_channel_index,
                                     num_events_returned, event_timestamps, event_values):
        """
        Retrieve digital even channel data
        
        Args:
            file_handle - file handle
            zero_based_channel_index - zero-based digital event channel index
            num_events_returned - c_ulonglong class instance
            event_timestamps - c_longlong class instance array the size of PL2DigitalChannelInfo.m_NumberOfEvents
            event_values - c_ushort class instance array the size of PL2DigitalChannelInfo.m_NumberOfEvents
        
        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetDigitalChannelData(c_int(file_handle),
                                                             c_int(zero_based_channel_index),
                                                             byref(num_events_returned),
                                                             byref(event_timestamps),
                                                             byref(event_values))

        return self.result

    def pl2_get_digital_channel_data_by_name(self, file_handle, channel_name, num_events_returned,
                                             event_timestamps, event_values):
        """
        Retrieve digital even channel data
        
        Args:
            file_handle - file handle
            channel_name - digital event channel name
            num_events_returned - c_ulonglong class instance
            event_timestamps - c_longlong class instance array the size of PL2DigitalChannelInfo.m_NumberOfEvents
            event_values - c_ushort class instance array the size of PL2DigitalChannelInfo.m_NumberOfEvents
        
        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetDigitalChannelDataByName(c_int(file_handle),
                                                                   channel_name.encode('ascii'),
                                                                   byref(num_events_returned),
                                                                   byref(event_timestamps),
                                                                   byref(event_values))

        return self.result

    def pl2_get_digital_channel_data_by_source(self, file_handle, source_id,
                                               one_based_channel_index_in_source,
                                               num_events_returned, event_timestamps,
                                               event_values):
        """
        Retrieve digital even channel data
        
        Args:
            file_handle - file handle
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source
            num_events_returned - c_ulonglong class instance
            event_timestamps - c_longlong class instance array the size of PL2DigitalChannelInfo.m_NumberOfEvents
            event_values - c_ushort class instance array the size of PL2DigitalChannelInfo.m_NumberOfEvents
        
        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetDigitalChannelDataBySource(c_int(file_handle),
                                                                     c_int(source_id), c_int(
                one_based_channel_index_in_source), byref(num_events_returned),
                                                                     byref(event_timestamps),
                                                                     byref(event_values))

        return self.result

    def pl2_get_start_stop_channel_info(self, file_handle, number_of_start_stop_events):
        """
        Retrieve information about start/stop channel
        
        Args:
            file_handle - file handle
            number_of_start_stop_events - c_ulonglong class instance
        
        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.result = c_int(0)
        self.result = self.pl2_dll.PL2_GetStartStopChannelInfo(c_int(file_handle),
                                                               byref(number_of_start_stop_events))

        return self.result

    def pl2_get_start_stop_channel_data(self, file_handle, num_events_returned, event_timestamps,
                                        event_values):
        """
        Retrieve digital channel data
        
        Args:
            file_handle - file handle
            num_events_returned - c_ulonglong class instance
            event_timestamps - c_longlong class instance
            event_values - point to c_ushort class instance
        
        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.result = c_int()
        self.result = self.pl2_dll.PL2_GetStartStopChannelData(c_int(file_handle),
                                                               byref(num_events_returned),
                                                               byref(event_timestamps),
                                                               byref(event_values))

        return self.result

    # PL2 data block functions purposefully not implemented.
    def pl2_read_first_data_block(self, file_handle):
        pass

    def pl2_read_next_data_block(self, file_handle):
        pass

    def pl2_get_data_block_info(self, file_handle):
        pass

    def pl2_get_data_block_timestamps(self, file_handle):
        pass

    def pl2_get_spike_data_block_units(self, file_handle):
        pass

    def pl2_get_spike_data_block_waveforms(self, file_handle):
        pass

    def pl2_get_analog_data_block_timestamp(self, file_handle):
        pass

    def pl2_get_analog_data_block_values(self, file_handle):
        pass

    def pl2_get_digital_data_block_timestamps(self, file_handle):
        pass

    def pl2_get_start_stop_data_block_timestamps(self, file_handle):
        pass

    def pl2_get_start_stop_data_block_values(self, file_handle):
        pass
