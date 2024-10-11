# pypl2lib.py - Classes and functions for accessing functions
# in PL2FileReader.dll
#
# (c) 2016 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

import platform
import subprocess
import pathlib
import warnings
import numpy as np

platform_is_windows = platform.system() == "Windows"

if platform_is_windows:
    import ctypes
else:
    is_wine_available = subprocess.run(["which", "wine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if is_wine_available.returncode != 0:
        raise ImportError("Wine is not installed. Please install wine to use the PL2FileReader.dll")

    from zugbruecke import CtypesSession

    if platform.system() == "Darwin":
        ctypes = CtypesSession(log_level=100, arch="win64")
    else:
        ctypes = CtypesSession(log_level=100)


class tm(ctypes.Structure):
    _fields_ = [
        ("tm_sec", ctypes.c_int),
        ("tm_min", ctypes.c_int),
        ("tm_hour", ctypes.c_int),
        ("tm_mday", ctypes.c_int),
        ("tm_mon", ctypes.c_int),
        ("tm_year", ctypes.c_int),
        ("tm_wday", ctypes.c_int),
        ("tm_yday", ctypes.c_int),
        ("tm_isdst", ctypes.c_int),
    ]


class PL2FileInfo(ctypes.Structure):
    _fields_ = [
        ("m_CreatorComment", ctypes.c_char * 256),
        ("m_CreatorSoftwareName", ctypes.c_char * 64),
        ("m_CreatorSoftwareVersion", ctypes.c_char * 16),
        ("m_CreatorDateTime", tm),
        ("m_CreatorDateTimeMilliseconds", ctypes.c_int),
        ("m_TimestampFrequency", ctypes.c_double),
        ("m_NumberOfChannelHeaders", ctypes.c_uint),
        ("m_TotalNumberOfSpikeChannels", ctypes.c_uint),
        ("m_NumberOfRecordedSpikeChannels", ctypes.c_uint),
        ("m_TotalNumberOfAnalogChannels", ctypes.c_uint),
        ("m_NumberOFRecordedAnalogChannels", ctypes.c_uint),
        ("m_NumberOfDigitalChannels", ctypes.c_uint),
        ("m_MinimumTrodality", ctypes.c_uint),
        ("m_MaximumTrodality", ctypes.c_uint),
        ("m_NumberOfNonOmniPlexSources", ctypes.c_uint),
        ("m_Unused", ctypes.c_int),
        ("m_ReprocessorComment", ctypes.c_char * 256),
        ("m_ReprocessorSoftwareName", ctypes.c_char * 64),
        ("m_ReprocessorSoftwareVersion", ctypes.c_char * 16),
        ("m_ReprocessorDateTime", tm),
        ("m_ReprocessorDateTimeMilliseconds", ctypes.c_int),
        ("m_StartRecordingTime", ctypes.c_ulonglong),
        ("m_DurationOfRecording", ctypes.c_ulonglong),
    ]


class PL2AnalogChannelInfo(ctypes.Structure):
    _fields_ = [
        ("m_Name", ctypes.c_char * 64),
        ("m_Source", ctypes.c_uint),
        ("m_Channel", ctypes.c_uint),
        ("m_ChannelEnabled", ctypes.c_uint),
        ("m_ChannelRecordingEnabled", ctypes.c_uint),
        ("m_Units", ctypes.c_char * 16),
        ("m_SamplesPerSecond", ctypes.c_double),
        ("m_CoeffToConvertToUnits", ctypes.c_double),
        ("m_SourceTrodality", ctypes.c_uint),
        ("m_OneBasedTrode", ctypes.c_ushort),
        ("m_OneBasedChannelInTrode", ctypes.c_ushort),
        ("m_NumberOfValues", ctypes.c_ulonglong),
        ("m_MaximumNumberOfFragments", ctypes.c_ulonglong),
    ]


class PL2SpikeChannelInfo(ctypes.Structure):
    _fields_ = [
        ("m_Name", ctypes.c_char * 64),
        ("m_Source", ctypes.c_uint),
        ("m_Channel", ctypes.c_uint),
        ("m_ChannelEnabled", ctypes.c_uint),
        ("m_ChannelRecordingEnabled", ctypes.c_uint),
        ("m_Units", ctypes.c_char * 16),
        ("m_SamplesPerSecond", ctypes.c_double),
        ("m_CoeffToConvertToUnits", ctypes.c_double),
        ("m_SamplesPerSpike", ctypes.c_uint),
        ("m_Threshold", ctypes.c_int),
        ("m_PreThresholdSamples", ctypes.c_uint),
        ("m_SortEnabled", ctypes.c_uint),
        ("m_SortMethod", ctypes.c_uint),
        ("m_NumberOfUnits", ctypes.c_uint),
        ("m_SortRangeStart", ctypes.c_uint),
        ("m_SortRangeEnd", ctypes.c_uint),
        ("m_UnitCounts", ctypes.c_ulonglong * 256),
        ("m_SourceTrodality", ctypes.c_uint),
        ("m_OneBasedTrode", ctypes.c_ushort),
        ("m_OneBasedChannelInTrode", ctypes.c_ushort),
        ("m_NumberOfSpikes", ctypes.c_ulonglong),
    ]


class PL2DigitalChannelInfo(ctypes.Structure):
    _fields_ = [
        ("m_Name", ctypes.c_char * 64),
        ("m_Source", ctypes.c_uint),
        ("m_Channel", ctypes.c_uint),
        ("m_ChannelEnabled", ctypes.c_uint),
        ("m_ChannelRecordingEnabled", ctypes.c_uint),
        ("m_NumberOfEvents", ctypes.c_ulonglong),
    ]


def to_array(c_array):
    return np.ctypeslib.as_array(c_array)


def to_array_nonzero(c_array):
    a = np.ctypeslib.as_array(c_array)
    return a[np.where(a)]


class PyPL2FileReader:
    def __init__(self, pl2_dll_file_path=None):
        """
        PyPL2FileReader class implements functions in the C++ PL2 File Reader
        API provided by Plexon, Inc.

        Args:
            pl2_dll_file_path - path where PL2FileReader.dll is location.
                The default value assumes the .dll files are located in the
                'bin' directory, which is a subdirectory of this package.
                Any file path passed is converted to an absolute path and checked
                to see if the .dll exists there.

        Returns:
            None
        """
        self._file_handle = ctypes.c_int(0)
        self.pl2_file_info = None
        if pl2_dll_file_path is None:
            pl2_dll_file_path = pathlib.Path(__file__).parents[1] / "bin" / "PL2FileReader.dll"
        self.pl2_dll_path = pathlib.Path(pl2_dll_file_path).absolute()

        # use default '32bit' dll version
        self.pl2_dll_file_path = self.pl2_dll_path

        try:
            self.pl2_dll = ctypes.CDLL(str(self.pl2_dll_file_path))
        except IOError:
            raise IOError(
                f"Error: Can't load PL2FileReader.dll at: {self.pl2_dll_file_path}. "
                "PL2FileReader.dll is bundled with the C++ PL2 Offline Files SDK "
                "located on the Plexon Inc website: www.plexon.com "
                "Contact Plexon Support for more information: support@plexon.com"
            )

    def pl2_open_file(self, pl2_file):
        """
        Opens and returns a handle to a PL2 file.

        Args:
            pl2_file - full path of the file

        Returns:
            None

        """
        if isinstance(pl2_file, pathlib.Path):
            pl2_file = str(pl2_file)
        self.pl2_dll.PL2_OpenFile.argtypes = (
            ctypes.POINTER(ctypes.c_char),
            ctypes.POINTER(ctypes.c_int),
        )
        self.pl2_dll.PL2_OpenFile.memsync = [
            {
                "p": [0],  # ctypes.POINTER argument
                "n": True,  # null-terminated string flag
            }
        ]

        self.pl2_dll.PL2_OpenFile(
            pl2_file.encode("ascii"),
            ctypes.byref(self._file_handle),
        )

        # load file info
        self.pl2_get_file_info()
        # check if spiking data can be loaded using zugbruecke
        self._check_spike_channel_data_consistency()

    def pl2_close_file(self):
        """
        Closes handle to PL2 file.

        Returns:
            None
        """

        self.pl2_dll.PL2_CloseFile.argtypes = (ctypes.POINTER(ctypes.c_int),)
        self.pl2_dll.PL2_CloseFile(ctypes.c_int(1))

    def pl2_close_all_files(self):
        """
        Closes all files that have been opened by the .dll

        Args:
            None

        Returns:
            None
        """

        self.pl2_dll.PL2_CloseAllFiles.argtypes = ()
        self.pl2_dll.PL2_CloseAllFiles()

    def pl2_get_last_error(self):
        """
        Retrieve description of the last error

        Returns:
            str - error message
        """

        self.pl2_dll.PL2_GetLastError.argtypes = (
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int,
        )
        self.pl2_dll.PL2_GetLastError.memsync = [{"p": [0], "l": [1], "t": ctypes.c_char}]

        buffer = (ctypes.c_char * 256)()
        self.pl2_dll.PL2_GetLastError(buffer, ctypes.c_int(256))

        return str(buffer.value)

    def _check_spike_channel_data_consistency(self):
        """
        Check if all spiking channels use the same number of samples per
        waveform. Only in this case can zugbruecke reliably load spiking data
        """

        if not self.pl2_file_info.m_TotalNumberOfSpikeChannels:
            return

        # extract samples per spike of first channel
        channel_info = self.pl2_get_spike_channel_info(0)
        n_samples_per_spike = channel_info.m_SamplesPerSpike

        # compare with all other channels
        for i in range(1, self.pl2_file_info.m_TotalNumberOfSpikeChannels):
            channel_info = self.pl2_get_spike_channel_info(i)

            if channel_info.m_SamplesPerSpike != n_samples_per_spike:
                warnings.warn(
                    "The spike channels contain different number of samples per spike. "
                    "Spiking data can probably not be loaded using zugbruecke. "
                    "Use a windows operating system or remove the offending channels "
                    "from the file."
                )
                return

    def pl2_get_file_info(self):
        """
        Retrieve information about pl2 file.

        Returns:
            pl2_file_info - PL2FileInfo class instance
        """

        self.pl2_file_info = PL2FileInfo()

        self.pl2_dll.PL2_GetFileInfo.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(PL2FileInfo),
        )

        result = self.pl2_dll.PL2_GetFileInfo(self._file_handle, ctypes.byref(self.pl2_file_info))

        # If res is 0, print error message
        if result == 0:
            self._print_error()
            return None

        return self.pl2_file_info

    def pl2_get_analog_channel_info(self, zero_based_channel_index):
        """
        Retrieve information about an analog channel

        Args:
            zero_based_channel_index - zero-based analog channel index

        Returns:
            pl2_analog_channel_info - PL2AnalogChannelInfo class instance
        """

        self.pl2_dll.PL2_GetAnalogChannelInfo.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(PL2AnalogChannelInfo),
        )

        pl2_analog_channel_info = PL2AnalogChannelInfo()
        result = self.pl2_dll.PL2_GetAnalogChannelInfo(
            self._file_handle, ctypes.c_int(zero_based_channel_index), ctypes.byref(pl2_analog_channel_info)
        )

        if not result:
            self._print_error()
            return None

        return pl2_analog_channel_info

    def pl2_get_analog_channel_info_by_name(self, channel_name):
        """
        Retrieve information about an analog channel

        Args:
            channel_name - analog channel name

        Returns:
            pl2_analog_channel_info - PL2AnalogChannelInfo class instance
        """

        self.pl2_dll.PL2_GetAnalogChannelInfoByName.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char),
            ctypes.POINTER(PL2AnalogChannelInfo),
        )

        self.pl2_dll.PL2_GetAnalogChannelInfoByName.memsync = [
            {
                "p": [1],
                "n": True,  # null-terminated string flag
            }
        ]

        if hasattr(channel_name, "encode"):
            channel_name = channel_name.encode("ascii")

        pl2_analog_channel_info = PL2AnalogChannelInfo()

        result = self.pl2_dll.PL2_GetAnalogChannelInfoByName(
            self._file_handle, channel_name, ctypes.byref(pl2_analog_channel_info)
        )

        if not result:
            self._print_error()
            return None
        return pl2_analog_channel_info

    def pl2_get_analog_channel_info_by_source(self, source_id, one_based_channel_index_in_source):
        """
        Retrieve information about an analog channel

        Args:
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source

        Returns:
            pl2_analog_channel_info - PL2AnalogChannelInfo class instance
        """

        self.pl2_dll.PL2_GetAnalogChannelInfoBySource.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(PL2AnalogChannelInfo),
        )

        pl2_analog_channel_info = PL2AnalogChannelInfo()
        result = self.pl2_dll.PL2_GetAnalogChannelInfoBySource(
            self._file_handle,
            ctypes.c_int(source_id),
            ctypes.c_int(one_based_channel_index_in_source),
            ctypes.byref(pl2_analog_channel_info),
        )

        if not result:
            self._print_error()
            return None

        return pl2_analog_channel_info

    def pl2_get_analog_channel_data(self, zero_based_channel_index):
        """
        Retrieve analog channel data

        Args:
            zero_based_channel_index - zero based channel index

        Returns:
            fragment_timestamps - array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            fragment_counts - array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            values - array the size of PL2AnalogChannelInfo.m_NumberOfValues
        """

        achannel_info = self.pl2_get_analog_channel_info(zero_based_channel_index)

        num_fragments_returned = ctypes.c_ulonglong(achannel_info.m_MaximumNumberOfFragments)
        num_data_points_returned = ctypes.c_ulonglong(achannel_info.m_NumberOfValues)
        fragment_timestamps = (ctypes.c_longlong * achannel_info.m_MaximumNumberOfFragments)()
        fragment_counts = (ctypes.c_ulonglong * achannel_info.m_MaximumNumberOfFragments)()
        values = (ctypes.c_short * achannel_info.m_NumberOfValues)()

        self.pl2_dll.PL2_GetAnalogChannelData.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_short),
        )

        self.pl2_dll.PL2_GetAnalogChannelData.memsync = [
            {"p": [4], "l": [2], "t": ctypes.c_longlong},
            {"p": [5], "l": [2], "t": ctypes.c_ulonglong},
            {"p": [6], "l": [3], "t": ctypes.c_short},
        ]

        result = self.pl2_dll.PL2_GetAnalogChannelData(
            self._file_handle,
            ctypes.c_int(zero_based_channel_index),
            num_fragments_returned,
            num_data_points_returned,
            fragment_timestamps,
            fragment_counts,
            values,
        )

        if not result:
            self._print_error()
            return None

        return to_array(fragment_timestamps), to_array(fragment_counts), to_array(values)

    def pl2_get_analog_channel_data_subset(self):
        pass

    def pl2_get_analog_channel_data_by_name(self, channel_name):
        """
        Retrieve analog channel data

        Args:
            channel_name - analog channel name

        Returns:
            fragment_timestamps - array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            fragment_counts - array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            values - array the size of PL2AnalogChannelInfo.m_NumberOfValues
        """

        if hasattr(channel_name, "encode"):
            channel_name = channel_name.encode("ascii")

        achannel_info = self.pl2_get_analog_channel_info_by_name(channel_name)

        num_fragments_returned = ctypes.c_ulonglong(achannel_info.m_MaximumNumberOfFragments)
        num_data_points_returned = ctypes.c_ulonglong(achannel_info.m_NumberOfValues)
        fragment_timestamps = (ctypes.c_longlong * achannel_info.m_MaximumNumberOfFragments)()
        fragment_counts = (ctypes.c_ulonglong * achannel_info.m_MaximumNumberOfFragments)()
        values = (ctypes.c_short * achannel_info.m_NumberOfValues)()

        self.pl2_dll.PL2_GetAnalogChannelDataByName.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_short),
        )

        self.pl2_dll.PL2_GetAnalogChannelDataByName.memsync = [
            {
                "p": [1],
                "n": True,  # null-terminated string flag
            },
            {"p": [4], "l": [2], "t": ctypes.c_longlong},
            {"p": [5], "l": [2], "t": ctypes.c_ulonglong},
            {"p": [6], "l": [3], "t": ctypes.c_short},
        ]

        result = self.pl2_dll.PL2_GetAnalogChannelDataByName(
            self._file_handle,
            channel_name,
            num_fragments_returned,
            num_data_points_returned,
            fragment_timestamps,
            fragment_counts,
            values,
        )

        if not result:
            self._print_error()
            return None

        return to_array(fragment_timestamps), to_array(fragment_counts), to_array(values)

    def pl2_get_analog_channel_data_by_source(self, source_id, one_based_channel_index_in_source):
        """
        Retrieve analog channel data

        Args:
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source

        Returns:
            fragment_timestamps - array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            fragment_counts - array the size of PL2AnalogChannelInfo.m_MaximumNumberOfFragments
            values - array the size of PL2AnalogChannelInfo.m_NumberOfValues
        """

        achannel_info = self.pl2_get_analog_channel_info_by_source(source_id, one_based_channel_index_in_source)

        num_fragments_returned = ctypes.c_ulonglong(achannel_info.m_MaximumNumberOfFragments)
        num_data_points_returned = ctypes.c_ulonglong(achannel_info.m_NumberOfValues)
        fragment_timestamps = (ctypes.c_longlong * achannel_info.m_MaximumNumberOfFragments)()
        fragment_counts = (ctypes.c_ulonglong * achannel_info.m_MaximumNumberOfFragments)()
        values = (ctypes.c_short * achannel_info.m_NumberOfValues)()

        self.pl2_dll.PL2_GetAnalogChannelDataBySource.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_short),
        )

        self.pl2_dll.PL2_GetAnalogChannelDataBySource.memsync = [
            {"p": [5], "l": [3], "t": ctypes.c_longlong},
            {"p": [6], "l": [3], "t": ctypes.c_ulonglong},
            {"p": [7], "l": [4], "t": ctypes.c_short},
        ]

        result = self.pl2_dll.PL2_GetAnalogChannelDataBySource(
            self._file_handle,
            ctypes.c_int(source_id),
            ctypes.c_int(one_based_channel_index_in_source),
            num_fragments_returned,
            num_data_points_returned,
            fragment_timestamps,
            fragment_counts,
            values,
        )

        if not result:
            self._print_error()
            return None

        return to_array(fragment_timestamps), to_array(fragment_counts), to_array(values)

    def pl2_get_spike_channel_info(self, zero_based_channel_index):
        """
        Retrieve information about a spike channel

        Args:
            zero_based_channel_index - zero-based spike channel index

        Returns:
            pl2_spike_channel_info - PL2SpikeChannelInfo class instance
        """

        pl2_spike_channel_info = PL2SpikeChannelInfo()

        self.pl2_dll.PL2_GetSpikeChannelInfo.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(PL2SpikeChannelInfo),
        )

        result = self.pl2_dll.PL2_GetSpikeChannelInfo(
            self._file_handle, ctypes.c_int(zero_based_channel_index), ctypes.byref(pl2_spike_channel_info)
        )

        if not result:
            self._print_error()
            return None

        return pl2_spike_channel_info

    def pl2_get_spike_channel_info_by_name(self, channel_name):
        """
        Retrieve information about a spike channel

        Args:
            channel_name - spike channel name

        Returns:
            pl2_spike_channel_info - PL2SpikeChannelInfo class instance
        """

        self.pl2_dll.PL2_GetSpikeChannelInfoByName.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char),
            ctypes.POINTER(PL2SpikeChannelInfo),
        )

        if hasattr(channel_name, "encode"):
            channel_name = channel_name.encode("ascii")

        self.pl2_dll.PL2_GetSpikeChannelInfoByName.memsync = [
            {
                "p": [1],
                "n": True,  # null-terminated string flag
            }
        ]

        pl2_spike_channel_info = PL2SpikeChannelInfo()

        result = self.pl2_dll.PL2_GetSpikeChannelInfoByName(
            self._file_handle, channel_name, ctypes.byref(pl2_spike_channel_info)
        )
        if not result:
            self._print_error()
            return None
        return pl2_spike_channel_info

    def pl2_get_spike_channel_info_by_source(self, source_id, one_based_channel_index_in_source):
        """
        Retrieve information about a spike channel

        Args:
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source

        Returns:
            pl2_spike_channel_info - PL2SpikeChannelInfo class instance
        """

        pl2_spike_channel_info = PL2SpikeChannelInfo()

        self.pl2_dll.PL2_GetSpikeChannelInfoBySource.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(PL2SpikeChannelInfo),
        )
        result = self.pl2_dll.PL2_GetSpikeChannelInfoBySource(
            self._file_handle,
            ctypes.c_int(source_id),
            ctypes.c_int(one_based_channel_index_in_source),
            ctypes.byref(pl2_spike_channel_info),
        )

        if not result:
            self._print_error()
            return None

        return pl2_spike_channel_info

    def pl2_get_spike_channel_data(self, zero_based_channel_index):
        """
        Retrieve spike channel data

        Args:
            zero_based_channel_index - zero based channel index

        Returns:
            spike_timestamps - array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            units - array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            values - array the size of (PL2SpikeChannelInfo.m_NumberOfSpikes * PL2SpikeChannelInfo.m_SamplesPerSpike)
        """

        # extracting m_SamplesPerSpike to prepare data reading
        # This solution only works if all channels have the same number of samples per spike
        # as ctypes / zugbruecke is caching the memsync attribute once defined once
        schannel_info = self.pl2_get_spike_channel_info(zero_based_channel_index)
        samples_per_spike = schannel_info.m_SamplesPerSpike

        self.pl2_dll.PL2_GetSpikeChannelData.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.POINTER(ctypes.c_short),
        )

        self.pl2_dll.PL2_GetSpikeChannelData.memsync = [
            {"p": [3], "l": [2], "t": ctypes.c_longlong},
            {"p": [4], "l": [2], "t": ctypes.c_ushort},
            {"p": [5], "l": ([2],), "func": f"lambda x: x.value * {samples_per_spike}", "t": ctypes.c_short},
        ]

        # These will be filled in by the dll method.
        num_spikes_returned = ctypes.c_ulonglong(schannel_info.m_NumberOfSpikes)
        spike_timestamps = (ctypes.c_ulonglong * schannel_info.m_NumberOfSpikes)()
        units = (ctypes.c_ushort * schannel_info.m_NumberOfSpikes)()
        values = (ctypes.c_short * (schannel_info.m_NumberOfSpikes * schannel_info.m_SamplesPerSpike))()

        result = self.pl2_dll.PL2_GetSpikeChannelData(
            self._file_handle,
            ctypes.c_int(zero_based_channel_index),
            num_spikes_returned,
            spike_timestamps,
            units,
            values,
        )

        if not result:
            self._print_error()
            return None

        spike_array_shape = (schannel_info.m_NumberOfSpikes, schannel_info.m_SamplesPerSpike)

        return to_array(spike_timestamps), to_array(units), to_array(values).reshape(spike_array_shape)

    def pl2_get_spike_channel_data_by_name(self, channel_name):
        """
        Retrieve spike channel data

        Args:
            channel_name = channel name

        Returns:
            spike_timestamps - array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            units - array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            values - array the size of (PL2SpikeChannelInfo.m_NumberOfSpikes * PL2SpikeChannelInfo.m_SamplesPerSpike)
        """

        if hasattr(channel_name, "encode"):
            channel_name = channel_name.encode("ascii")

        self.pl2_dll.PL2_GetSpikeChannelDataByName.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.POINTER(ctypes.c_short),
        )

        # extracting m_SamplesPerSpike to prepare data reading
        # This solution only works if all channels have the same number of samples per spike
        # as ctypes / zugbruecke is caching the memsync attribute once defined once
        schannel_info = self.pl2_get_spike_channel_info_by_name(channel_name)
        samples_per_spike = schannel_info.m_SamplesPerSpike

        self.pl2_dll.PL2_GetSpikeChannelDataByName.memsync = [
            {
                "p": [1],
                "n": True,  # null-terminated string flag
            },
            {"p": [3], "l": [2], "t": ctypes.c_longlong},
            {"p": [4], "l": [2], "t": ctypes.c_ushort},
            {"p": [5], "l": ([2],), "func": f"lambda x: x.value * {samples_per_spike}", "t": ctypes.c_short},
        ]

        # These will be filled in by the dll method.
        num_spikes_returned = ctypes.c_ulonglong(schannel_info.m_NumberOfSpikes)
        spike_timestamps = (ctypes.c_ulonglong * schannel_info.m_NumberOfSpikes)()
        units = (ctypes.c_ushort * schannel_info.m_NumberOfSpikes)()
        values = (ctypes.c_short * (schannel_info.m_NumberOfSpikes * schannel_info.m_SamplesPerSpike))()

        result = self.pl2_dll.PL2_GetSpikeChannelDataByName(
            self._file_handle, channel_name, num_spikes_returned, spike_timestamps, units, values
        )

        if not result:
            self._print_error()
            return None

        spike_array_shape = (schannel_info.m_NumberOfSpikes, schannel_info.m_SamplesPerSpike)

        return to_array(spike_timestamps), to_array(units), to_array(values).reshape(spike_array_shape)

    def pl2_get_spike_channel_data_by_source(self, source_id, one_based_channel_index_in_source):
        """
        Retrieve spike channel data

        Args:
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source

        Returns:
            spike_timestamps - array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            units - array the size of PL2SpikeChannelInfo.m_NumberOfSpikes
            values - array the size of (PL2SpikeChannelInfo.m_NumberOfSpikes * PL2SpikeChannelInfo.m_SamplesPerSpike)
        """

        self.pl2_dll.PL2_GetSpikeChannelDataBySource.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.POINTER(ctypes.c_short),
        )

        # extracting m_SamplesPerSpike to prepare data reading
        # This solution only works if all channels have the same number of samples per spike
        # as ctypes / zugbruecke is caching the memsync attribute once defined once
        schannel_info = self.pl2_get_spike_channel_info_by_source(source_id, one_based_channel_index_in_source)
        samples_per_spike = schannel_info.m_SamplesPerSpike

        self.pl2_dll.PL2_GetSpikeChannelDataBySource.memsync = [
            {"p": [4], "l": [3], "t": ctypes.c_longlong},
            {"p": [5], "l": [3], "t": ctypes.c_ushort},
            {"p": [6], "l": ([3],), "func": f"lambda x: x.value * {samples_per_spike}", "t": ctypes.c_short},
        ]

        # These will be filled in by the dll method.
        num_spikes_returned = ctypes.c_ulonglong(schannel_info.m_NumberOfSpikes)
        spike_timestamps = (ctypes.c_ulonglong * schannel_info.m_NumberOfSpikes)()
        units = (ctypes.c_ushort * schannel_info.m_NumberOfSpikes)()
        values = (ctypes.c_short * (schannel_info.m_NumberOfSpikes * schannel_info.m_SamplesPerSpike))()

        result = self.pl2_dll.PL2_GetSpikeChannelDataBySource(
            self._file_handle,
            ctypes.c_int(source_id),
            ctypes.c_int(one_based_channel_index_in_source),
            num_spikes_returned,
            spike_timestamps,
            units,
            values,
        )

        if not result:
            self._print_error()
            return None

        spike_array_shape = (schannel_info.m_NumberOfSpikes, schannel_info.m_SamplesPerSpike)

        return to_array(spike_timestamps), to_array(units), to_array(values).reshape(spike_array_shape)

    def pl2_get_digital_channel_info(self, zero_based_channel_index):
        """
        Retrieve information about a digital event channel

        Args:
            zero_based_channel_index - zero-based digital event channel index

        Returns:
            pl2_digital_channel_info - PL2DigitalChannelInfo class instance
        """

        self.pl2_dll.PL2_GetDigitalChannelInfo.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(PL2DigitalChannelInfo),
        )

        pl2_digital_channel_info = PL2DigitalChannelInfo()

        result = self.pl2_dll.PL2_GetDigitalChannelInfo(
            self._file_handle, ctypes.c_int(zero_based_channel_index), ctypes.byref(pl2_digital_channel_info)
        )

        if not result:
            self._print_error()
            return None

        return pl2_digital_channel_info

    def pl2_get_digital_channel_info_by_name(self, channel_name):
        """
        Retrieve information about a digital event channel

        Args:
            channel_name - digital event channel name

        Returns:
            pl2_digital_channel_info - PL2DigitalChannelInfo class instance
        """

        if hasattr(channel_name, "encode"):
            channel_name = channel_name.encode("ascii")

        self.pl2_dll.PL2_GetDigitalChannelInfoByName.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char),
            ctypes.POINTER(PL2DigitalChannelInfo),
        )

        self.pl2_dll.PL2_GetDigitalChannelInfoByName.memsync = [
            {
                "p": [1],  # ctypes.POINTER argument
                "n": True,  # null-terminated string flag
            }
        ]

        pl2_digital_channel_info = PL2DigitalChannelInfo()

        result = self.pl2_dll.PL2_GetDigitalChannelInfoByName(
            self._file_handle, channel_name, ctypes.byref(pl2_digital_channel_info)
        )

        if not result:
            self._print_error()
            return None

        return pl2_digital_channel_info

    def pl2_get_digital_channel_info_by_source(self, source_id, one_based_channel_index_in_source):
        """
        Retrieve information about a digital event channel

        Args:
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source

        Returns:
            pl2_digital_channel_info - PL2DigitalChannelInfo class instance
        """

        self.pl2_dll.PL2_GetDigitalChannelInfoBySource.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(PL2DigitalChannelInfo),
        )

        pl2_digital_channel_info = PL2DigitalChannelInfo()

        result = self.pl2_dll.PL2_GetDigitalChannelInfoBySource(
            self._file_handle,
            ctypes.c_int(source_id),
            ctypes.c_int(one_based_channel_index_in_source),
            ctypes.byref(pl2_digital_channel_info),
        )

        if not result:
            self._print_error()
            return None

        return pl2_digital_channel_info

    def pl2_get_digital_channel_data(self, zero_based_channel_index):
        """
        Retrieve digital even channel data

        Args:
            zero_based_channel_index - zero-based digital event channel index

        Returns:
            event_timestamps - array the size of PL2DigitalChannelInfo.m_NumberOfEvents
            event_values - array the size of PL2DigitalChannelInfo.m_NumberOfEvents
        """

        self.pl2_dll.PL2_GetDigitalChannelData.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_short),
        )

        self.pl2_dll.PL2_GetDigitalChannelData.memsync = [
            {"p": [3], "l": [2], "t": ctypes.c_longlong},
            {"p": [4], "l": [2], "t": ctypes.c_ushort},
        ]

        echannel_info = self.pl2_get_digital_channel_info(zero_based_channel_index)

        # These will be filled in by the dll method.
        num_events_returned = ctypes.c_ulonglong(echannel_info.m_NumberOfEvents)
        event_timestamps = (ctypes.c_longlong * echannel_info.m_NumberOfEvents)()
        event_values = (ctypes.c_ushort * echannel_info.m_NumberOfEvents)()

        result = self.pl2_dll.PL2_GetDigitalChannelData(
            self._file_handle,
            ctypes.c_int(zero_based_channel_index),
            num_events_returned,
            event_timestamps,
            event_values,
        )

        if not result:
            self._print_error()
            return None

        return to_array(event_timestamps), to_array(event_values)

    def pl2_get_digital_channel_data_by_name(self, channel_name):
        """
        Retrieve digital even channel data

        Args:
            channel_name - digital event channel name

        Returns:
            event_timestamps - array the size of PL2DigitalChannelInfo.m_NumberOfEvents
            event_values - array the size of PL2DigitalChannelInfo.m_NumberOfEvents
        """

        if hasattr(channel_name, "encode"):
            channel_name = channel_name.encode("ascii")

        self.pl2_dll.PL2_GetDigitalChannelDataByName.argtypes = (
            ctypes.c_int,  # file handle
            ctypes.POINTER(ctypes.c_char),  # channel name
            ctypes.POINTER(ctypes.c_ulonglong),  # equivalent to m_NumberOfEvents
            ctypes.POINTER(ctypes.c_longlong),  # array of timestamps of length m_NumberOfEvents
            ctypes.POINTER(ctypes.c_ushort),  # array of values m_NumberOfEvents
        )

        self.pl2_dll.PL2_GetDigitalChannelDataByName.memsync = [
            {
                "p": [1],  # ctypes.POINTER argument
                "n": True,  # null-terminated string flag
            },
            {"p": [3], "l": [2], "t": ctypes.c_longlong},
            {"p": [4], "l": [2], "t": ctypes.c_ushort},
        ]

        echannel_info = self.pl2_get_digital_channel_info_by_name(channel_name)

        # These will be filled in by the dll method.
        num_events_returned = ctypes.c_ulonglong(echannel_info.m_NumberOfEvents)
        event_timestamps = (ctypes.c_longlong * echannel_info.m_NumberOfEvents)()
        event_values = (ctypes.c_ushort * echannel_info.m_NumberOfEvents)()

        result = self.pl2_dll.PL2_GetDigitalChannelDataByName(
            self._file_handle, channel_name, num_events_returned, event_timestamps, event_values
        )

        if not result:
            self._print_error()
            return None

        return to_array(event_timestamps), to_array(event_values)

    def pl2_get_digital_channel_data_by_source(self, source_id, one_based_channel_index_in_source):
        """
        Retrieve digital even channel data

        Args:
            source_id - numeric source ID
            one_based_channel_index_in_source - one-based channel index within the source

        Returns:
            event_timestamps - array the size of PL2DigitalChannelInfo.m_NumberOfEvents
            event_values - array the size of PL2DigitalChannelInfo.m_NumberOfEvents
        """

        self.pl2_dll.PL2_GetDigitalChannelDataBySource.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_ushort),
        )

        self.pl2_dll.PL2_GetDigitalChannelDataBySource.memsync = [
            {"p": [4], "l": [3], "t": ctypes.c_longlong},
            {"p": [5], "l": [3], "t": ctypes.c_ushort},
        ]

        echannel_info = self.pl2_get_digital_channel_info_by_source(source_id, one_based_channel_index_in_source)

        # These will be filled in by the dll method.
        num_events_returned = ctypes.c_ulonglong(echannel_info.m_NumberOfEvents)
        event_timestamps = (ctypes.c_longlong * echannel_info.m_NumberOfEvents)()
        event_values = (ctypes.c_ushort * echannel_info.m_NumberOfEvents)()

        result = self.pl2_dll.PL2_GetDigitalChannelDataBySource(
            self._file_handle,
            ctypes.c_int(source_id),
            ctypes.c_int(one_based_channel_index_in_source),
            num_events_returned,
            event_timestamps,
            event_values,
        )

        if not result:
            self._print_error()
            return None

        return to_array(event_timestamps), to_array(event_values)

    def pl2_get_start_stop_channel_info(self, number_of_start_stop_events):
        """
        Retrieve information about start/stop channel

        Args:
            _file_handle - file handle
            number_of_start_stop_events - ctypes.c_ulonglong class instance

        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.pl2_dll.PL2_GetStartStopChannelInfo.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong))

        result = self.pl2_dll.PL2_GetStartStopChannelInfo(self._file_handle, number_of_start_stop_events)

        return result

    def pl2_get_start_stop_channel_data(self, num_events_returned, event_timestamps, event_values):
        """
        Retrieve digital channel data

        Args:
            _file_handle - file handle
            num_events_returned - ctypes.c_ulonglong class instance
            event_timestamps - ctypes.c_longlong class instance
            event_values - point to ctypes.c_ushort class instance

        Returns:
            1 - Success
            0 - Failure
            The class instances passed to the function are filled with values
        """

        self.pl2_dll.PL2_GetStartStopChannelInfo.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_ushort),
        )

        self.pl2_dll.PL2_GetStartStopChannelInfo.memsync = [
            {"p": [2], "l": [1], "t": ctypes.c_longlong},
            {"p": [3], "l": [1], "t": ctypes.c_ushort},
        ]

        result = self.pl2_dll.PL2_GetStartStopChannelData(
            self._file_handle, num_events_returned, event_timestamps, event_values
        )

        return result

    def _print_error(self):
        error_message = self.pl2_get_last_error()
        print(f"pypl2lib error: {error_message}")

    # PL2 data block functions purposefully not implemented.
    def pl2_read_first_data_block(self):
        pass

    def pl2_read_next_data_block(self):
        pass

    def pl2_get_data_block_info(self):
        pass

    def pl2_get_data_block_timestamps(self):
        pass

    def pl2_get_spike_data_block_units(self):
        pass

    def pl2_get_spike_data_block_waveforms(self):
        pass

    def pl2_get_analog_data_block_timestamp(self):
        pass

    def pl2_get_analog_data_block_values(self):
        pass

    def pl2_get_digital_data_block_timestamps(self):
        pass

    def pl2_get_start_stop_data_block_timestamps(self):
        pass

    def pl2_get_start_stop_data_block_values(self):
        pass
