# coding=utf-8
"""
Module for reading binary files in the Blackrock format.
Authors: Michael Denker, Lyuba Zehl

blackrockio
===========

Classes
-------

BlackrockIO    - class to enable reading of BlackrockIO files
"""

import struct
import os
import datetime
import numpy as np
import quantities as pq
import neo
from neo.io.baseio import BaseIO


if __name__ == '__main__':
    pass


class BlackrockIO(BaseIO):
    """
    Class for reading data in a data files recorded by the Blackrock (Cerebus)
    recording system.

    Upon initialization, the class is linked to a set of Blackrock files. In this
    process, basic information about the file(s) is obtained, and stored in the
    class attributes. Data can be read as a neo Block object using the read_block
    method.

    Note: This routine will NOT handle files according to specification 2.3,
    where recording may have gaps (paused recording)!

    Inherits from:
            neo.io.BaseIO

    Attributes:
        associated (bool):
            True if the object is successfully associated to a set of Blackrock files
        associated_fileset (string):
            Name of the associated file set.
        nev_fileprefix (string):
            File name of the requested .nev file (without extension).
        nsx_fileprefix (string):
            File name of the requested .nsX file(s) (without extensions).
        nsx_avail (list):
            A list of all integers X for which the requested .nsX files exists.
        nev_avail (bool):
            True if the requested .nev file exists.
        num_channels_nsx (list):
            A list of integers of length ten. The entry at index X
            corresponds to the number of channels in the file .nsX.
        channel_id_nsx (list):
            This list contains at entry X a list of available channel IDs in the
            .nsX file.
        channel_id_nev (list):
            List of channel IDs on which spike events are detected.
        nsx_unit (list):
            List of compound Quantity objects that yield the time unit of the
            analog signal. The entry at index X relates to the .nsX file.
        nev_unit (Quantity):
            This compound Quantity is the unit of spike time stamps.
        waveform_unit (Quantity):
            This compound Quantity is the time unit of the sampled spike
            waveforms.
        analog_res (list):
            List of sampling frequencies in Hz of the analog data. The entry at
            position X corresponds to sampling frequency in the .nsX file.
        timestamp_res (int):
            Sampling frequency of events (spike times and markers) in Hz.
        waveform_res (int):
            Sampling frequency of spike waveforms in Hz.
        parameters_nsx (list):
            List of dictionaries of information extracted from .nsX files. The
            entry at position X corresponds to the information in the .nsX file.
        parameters_nev (dict):
            Dictionary of information extracted from .nev file.
        parameters_electrodes (list):
            List of dictionaries of information extracted from .nev file per
            electrode. The entry at position X corresponds to the information on
            electrode X (X=1-255; entry X=0 contains an empty dict).
        parameters_patient (dict):
            Dictionary of information extracted from .sif file.
        nev_content (string):
            Gives a textual description of the nev-file content.
        nsx_content (dict):
            For each file extension (keys), gives a textual description of the file content.
    """

    # Class variables demonstrating capabilities of this IO
    is_readable = True  # This a only reading class
    is_writable = False  # write is not supported

    # This IO can only manipulate continuous data, spikes, and events
    supported_objects = [neo.Block, neo.Segment, neo.AnalogSignal, neo.SpikeTrain, neo.EventArray, neo.RecordingChannelGroup, neo.RecordingChannel]

    # Keep things simple by always returning a block
    readable_objects = [neo.Block]

    # And write a block
    writeable_objects = []

    # Not sure what these do, if anything
    has_header = False
    is_streameable = False
    read_params = {}
    write_params = {}

    # The IO name and the file extensions it uses
    # Analog files are stored in files .ns0 - .ns6, spikes and events are stored in .nev
    #                    .nev - spike- and event-data; 30000 Hz
    #                    .ns6 - 30000 Hz (no digital filter)
    #                    .ns5 - 30000 Hz
    #                    .ns4 - 10000 Hz
    #                    .ns3 -  2000 Hz
    #                    .ns2 -  1000 Hz
    #                    .ns1 -   500 Hz
    name = 'Blackrock'
    description = 'This IO reads .nev/.nsX file of the Blackrock (Cerebus) recordings system ("Utah" array).'

    extensions = ['ns' + str(_) for _ in range(1, 7)]
    extensions.append('nev')

    # Operates on .nev and .nsX files
    mode = 'file'

    # Textual description of file types (extensions)
    # Only those nsX files are listed which are currently documented by Blackrock
    # Systems, i.e., ns1-ns6
    nev_content = "spike- and event-data; 30000 Hz"
    nsx_content = {"ns1": "analog data: 500 Hz",
                   "ns2": "analog data: 1000 Hz",
                   "ns3": "analog data: 2000 Hz",
                   "ns4": "analog data: 10000 Hz",
                   "ns5": "analog data: 30000 Hz",
                   "ns6": "analog data: 30000 Hz (no digital filter)"}



    def __init__(self, filename, nsx_override=None, nev_override=None, print_diagnostic=False) :
        """Initialize the BlackrockSession class and associate it to a file set.

        The Blackrock data format consists not of a single file, but a mixture
        of different files. This constructor associates itself with a set of
        files that constitute a common data set. During this association,
        several features (e.g., unit IDs) are already read from these data files
        for later use.

        Args:
            filename (string):
                Name of a Blackrock file set to associate with. The file
                extension(s) should not be included. This name is used as default
                filename for .nsX and .nev files.
            nsx_override (string):
                File name of the .nsX files (without extension). If None,
                filename is used.
                Default: None.
            nev_override (string):
                File name of the .nev files (without extension). If None,
                filename is used.
                Default: None.
            print_diagnostic (boolean):
                If true, the class will output diagnostic information. This
                option may be removed in the future!
                Default: False

        Returns:
            -

        Examples:
            a=BlackrockIO('myfile')

                Loads a set of file consisting of files myfile.nev, myfile.ns0, myfile.ns1,...,myfile.ns9


            b=BlackrockIO('myfile',nev_override='sorted')

                Loads the analog data from the set of files
                myfile.ns0,myfile.ns1,...,myfile.ns9,
                but reads spike/event data from sorted.nev.
        """

        self.associated = False
        # Call base constructor
        BaseIO.__init__(self)
        # Remember choice whether to print diagnostic messages or not
        self._print_diagnostic = print_diagnostic
        # Associate to the session
        self._associate(sessionname=filename, nsx_override=nsx_override, nev_override=nev_override)
        # For consistency with baseio
        self.filename = filename


    def _diagnostic_print(self, text):
        '''
        Print a diagnostic message.

        Args:
            text (string):
                Diagnostic text to print.

        Returns:
            -
        '''

        if self._print_diagnostic:
            print 'BlackrockIO: ' + text


    def __read_nsx_header(self, filehandle, nsx):
        '''
        Reads the .nsX basic header block and stores the information in the
        object's parameters_nsx dictionary.

        Args:
            filehandle (file object):
                Handle to the already opened .nsX file. The file pointer is
                expected to be at the starting position of the header block.
            nsx (int):
                Integer ID x of the .nsX file under consideration. Determines in
                which parameters_nsx dictionary the header information is
                read.

        Returns:
            -
        '''

        # Read header block ID
        fileheader = filehandle.read(8)

        # Determine if old or new file format version based on header ID
        if fileheader == 'NEURALSG':
            # This is a version 2.1 file
            self.parameters_nsx[nsx]['Version'] = '2.1'
            self.parameters_nsx[nsx]['VersionMajor'] = 2
            self.parameters_nsx[nsx]['VersionMinor'] = 1

            # Read label string and remove 0's
            dummy = filehandle.read(16)
            self.parameters_nsx[nsx]['Label'] = dummy.replace('\x00', '')

            # Read sampling frequency in Hz
            # Note: 30000/N is always an integer
            # .ns1: 500Hz, .ns2 1kHz, .ns3 2kHz, .ns4 10kHz, .ns5 30kHz
            # unsigned int 32bit, little endian
            (self.analog_res[nsx],) = struct.unpack('<I', filehandle.read(4))
            self.analog_res[nsx] = 30000 / self.analog_res[nsx]
            self.parameters_nsx[nsx]['AnalogTimeResolution'] = self.analog_res[nsx]

            # Read number of channels in file
            # unsigned int 32bit, little endian
            (self.num_channels_nsx[nsx],) = struct.unpack('<I', filehandle.read(4))
            self.parameters_nsx[nsx]['NumChannels'] = self.num_channels_nsx[nsx];

            # Read list of channel IDs
            # unsigned int 32bit, little endian
            self.channel_id_nsx[nsx] = struct.unpack('<' + str(self.num_channels_nsx[nsx]) + 'I', filehandle.read(4 * self.num_channels_nsx[nsx]))
            self.parameters_nsx[nsx]['ChannelIDs'] = list(self.channel_id_nsx[nsx])

            # Get file position
            self.__file_nsx_header_end_pos[nsx] = filehandle.tell()

        elif fileheader == 'NEURALCD':
            # Read version number >2.1
            # unsigned byte 8bit, little endian
            (major,) = struct.unpack('<B', filehandle.read(1))
            (minor,) = struct.unpack('<B', filehandle.read(1))
            self.parameters_nsx[nsx]['Version'] = str(major) + '.' + str(minor);
            self.parameters_nsx[nsx]['VersionMajor'] = major
            self.parameters_nsx[nsx]['VersionMinor'] = minor

            # Read length of header
            # unsigned int 32bit, little endian
            (self.__file_nsx_header_end_pos[nsx],) = struct.unpack('<I', filehandle.read(4))

            # Read label string and remove 0's
            self.parameters_nsx[nsx]['Label'] = filehandle.read(16).split('\x00', 1)[0]

            # Read comments and remove 0's
            self.parameters_nsx[nsx]['Comments'] = filehandle.read(256).split('\x00', 1)[0]

            # Read sampling frequency in Hz
            # Note: 30000/N is always an integer
            # .ns1: 500Hz, .ns2 1kHz, .ns3 2kHz, .ns4 10kHz, .ns5 30kHz
            # unsigned int 32bit, little endian
            (self.analog_res[nsx],) = struct.unpack('<I', filehandle.read(4))
            self.analog_res[nsx] = 30000 / self.analog_res[nsx]
            self.parameters_nsx[nsx]['AnalogTimeResolution'] = self.analog_res[nsx]

            # Read resolution
            # unsigned int 32bit, little endian
            (self.parameters_nsx[nsx]['EventTimeResolution'],) = struct.unpack('<I', filehandle.read(4))

            # Read date and time
            # 8 unsigned short 16bit, little endian
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            date_time = struct.unpack('<8H', filehandle.read(16))
            self.parameters_nsx[nsx]['Date'] = str(date_time[1]) + '/' + str(date_time[3]) + '/' + str(date_time[0]) + ' ' + weekdays[date_time[2] - 1]
            self.parameters_nsx[nsx]['Time'] = str(date_time[4]) + ':' + str(date_time[5]) + ':' + str(date_time[6]) + ':' + str(date_time[7])

            # Read number of channels in file
            # unsigned int 32bit, little endian
            (self.num_channels_nsx[nsx],) = struct.unpack('<I', filehandle.read(4))
            self.parameters_nsx[nsx]['NumChannels'] = self.num_channels_nsx[nsx];

            # Pre-allocate channel list
            self.channel_id_nsx[nsx] = self.num_channels_nsx[nsx] * [None]
            self.parameters_nsx[nsx]['ChannelLabel'] = []

            for channel_i in range(self.num_channels_nsx[nsx]):
                if filehandle.read(2) != 'CC':
                    raise IOError("Expected channel block " + str(channel_i) + "in .nsX header, but did not encounter correct block ID.")

                # Read list of channel IDs
                # unsigned short 16bit, little endian
                (self.channel_id_nsx[nsx][channel_i],) = struct.unpack('<H', filehandle.read(2))

                # Read label string and remove 0's
                # TODO: Transform this into a real dict
                self.parameters_nsx[nsx]['ChannelLabel'].append(filehandle.read(16).replace('\x00', ''))

                # TODO: Add info from CC ext header!
                # 47 bytes ignored for now
                dummy = filehandle.read(46)

            # Save channel ID list to parameters_nsx
            self.parameters_nsx[nsx]['ChannelIDs'] = self.channel_id_nsx[nsx]

            # Reading should now be at the position specified in the header
            if self.__file_nsx_header_end_pos[nsx] != filehandle.tell():
                raise IOError("Inconsistency while reading .nsX header.")

        else:
            # Unrecognized file format version
            raise IOError("Unrecognized file specification in .nsX file.")


    def __read_nev_header(self, filehandle):
        '''
        Reads the .nev basic header block and stores the information in the
        object's parameters_nev dictionary.

        Args:
            filehandle (file object):
                Handle to the already opened NEV file. The file pointer is
                expected to be at the starting position of the header block.

        Returns:
            -
        '''

        # Read header block ID
        fileheader = filehandle.read(8)

        # Read version number
        # unsigned byte 8bit, little endian
        (major,) = struct.unpack('<B', filehandle.read(1))
        (minor,) = struct.unpack('<B', filehandle.read(1))
        self.parameters_nev['Version'] = str(major) + '.' + str(minor);
        self.parameters_nev['VersionMajor'] = major
        self.parameters_nev['VersionMinor'] = minor

        if fileheader == 'NEURALEV':
            # Version 2.2+ may have a sif file containing experimenter information
            filename_sif = self.parameters_nev['SessionName'][0:-4] + '.sif'
            if os.path.exists(filename_sif):
                self._diagnostic_print("Note: Scanning " + filename_sif)

                # Load sif data
                filehandle_sif = open(filename_sif, 'r')
                dummy = str(filehandle_sif.readlines()).strip()
                self.parameters_patient['Institution'] = dummy[2]
                self.parameters_patient['Patient'] = dummy[5]
                filehandle_sif.close()
            else:
                self._diagnostic_print("No .sif file was found for session " + self.associated_fileset + ".")

        elif self.parameters_nev['Version'] != '2.1':
            raise IOError("Unrecognized .nev file specification.")

        # Read flags, currently basically unused
        # unsigned short 16bit, little endian
        (flags,) = struct.unpack('<H', filehandle.read(2))
        self.parameters_nev['Flags'] = hex(flags)

        # Read end position of extended headers, i.e. index of first data packet
        # unsigned long 32bit, little endian
        (self.__file_nev_ext_header_end_pos,) = struct.unpack('<I', filehandle.read(4))

        # Read number of packet bytes, i.e. byte per timestamp
        # unsigned long 32bit, little endian
        (self.__packet_bytes,) = struct.unpack('<I', filehandle.read(4))

        # Read time resolution in Hz of time stamps, i.e. data packets
        # unsigned long 32bit, little endian
        (self.timestamp_res,) = struct.unpack('<I', filehandle.read(4))
        self.parameters_nev['EventTimeResolution'] = self.timestamp_res

        # Read sampling frequency of waveforms in Hz
        # unsigned long 32bit, little endian
        (self.waveform_res,) = struct.unpack('<I', filehandle.read(4))
        self.parameters_nev['WaveformTimeResolution'] = self.waveform_res

        # Read date and time
        # 8 unsigned short 16bit, little endian
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        date_time = struct.unpack('<8H', filehandle.read(16))
        self.parameters_nev['Date'] = str(date_time[1]) + '/' + str(date_time[3]) + '/' + str(date_time[0]) + ' ' + weekdays[date_time[2] - 1]
        self.parameters_nev['Time'] = str(date_time[4]) + ':' + str(date_time[5]) + ':' + str(date_time[6]) + ':' + str(date_time[7])
        self.parameters_nev['DateTime'] = datetime.datetime(year=date_time[0], month=date_time[1], day=date_time[3],
                                                            hour=date_time[4], minute=date_time[5], second=date_time[6], microsecond=date_time[7])

        # Read comments and remove 0's
        self.parameters_nev['Application'] = filehandle.read(32).split('\x00', 1)[0]

        # Read comments and remove 0's
        self.parameters_nev['Comments'] = filehandle.read(256).split('\x00', 1)[0]

        # Read number of extended headers
        # unsigned long 32bit, little endian
        (self.__num_ext_header,) = struct.unpack('<I', filehandle.read(4))


    def __read_nev_ext_header(self, filehandle):
        '''
        Reads the extended header block and stores the information in the
        object's parameters_nev and parameters_electrodes dictionaries.

        Args:
            filehandle (file object):
                Handle to the already opened .nev file. The file pointer is
                expected to be at the starting position of the first header, and
                the variable self.__num_ext_header should contain the total number of
                extended headers from the basic header block.

        Returns:
            -
        '''

        # Initialize dictionary to hold electrode information...
        self.parameters_electrodes = [ {} for dummy in xrange(256) ]
        # ...and the number of byte per spike waveform sample, default to 1
        self.__byte_per_waveform_sample = 256 * [ 1 ]

        for _ext_block_i in range(0, self.__num_ext_header):
            # Read extended header block ID
            fileextheader = filehandle.read(8)
            if fileextheader == 'ARRAYNME':
                # Read name of electrode array and remove 0's
                self.parameters_nev['ArrayName'] = filehandle.read(24).replace('\x00', '')

            elif fileextheader == 'ECOMMENT':
                # Read extended comment and remove 0's
                self.parameters_nev['ExtendedComment'] = filehandle.read(24).replace('\x00', '')

            elif fileextheader == 'CCOMMENT':
                # Read continued comment and remove 0's
                self.parameters_nev["ExtendedComment"] = self.parameters_nev['ExtendedComment'] + filehandle.read(24).replace('\x00', '')

            elif fileextheader == 'MAPFILE':
                # Read file name of map file and remove 0's
                self.parameters_nev["MapFile"] = filehandle.read(24).replace('\x00', '')

            elif fileextheader == 'NEUEVWAV':
                # Read electrode number
                # unsigned short 16bit, little endian
                (el_id,) = struct.unpack('<H', filehandle.read(2))

                # Read physical connector (bank A-D)
                # unsigned byte 8bit, little endian
                (self.parameters_electrodes[el_id]['ConnectorID'],) = struct.unpack('<B', filehandle.read(1))

                # Read connector pin (pin number on connector, 1-37)
                # unsigned byte 8bit, little endian
                (self.parameters_electrodes[el_id]['ConnectorPin'],) = struct.unpack('<B', filehandle.read(1))

                # Read digitization factor in nV per LSB step
                # unsigned short 16bit, little endian
                (self.parameters_electrodes[el_id]['DigitizationFactor'],) = struct.unpack('<H', filehandle.read(2))

                # Read energy threshold in nV per LSB step
                # unsigned short 16bit, little endian
                (self.parameters_electrodes[el_id]['EnergyThreshold'],) = struct.unpack('<H', filehandle.read(2))

                # Read high threshold in µV
                # signed short 16bit, little endian
                (self.parameters_electrodes[el_id]['AmpThresholdHi'],) = struct.unpack('<h', filehandle.read(2))

                # Read low threshold in µV
                # signed short 16bit, little endian
                (self.parameters_electrodes[el_id]['AmpThresholdLo'],) = struct.unpack('<h', filehandle.read(2))

                # Read number of sorted units
                # unsigned byte 8bit, little endian
                (self.parameters_electrodes[el_id]['NumSortedUnits'],) = struct.unpack('<B', filehandle.read(1))

                # Read number of bytes per waveform sample
                # 0 or 1 both imply 1 byte, so convert a 0 to 1 for simplification
                # unsigned byte 8bit, little endian
                (self.parameters_electrodes[el_id]['NumBytesPerWaveform'],) = struct.unpack('<B', filehandle.read(1))
                self.parameters_electrodes[el_id]['NumBytesPerWaveform'] += (self.parameters_electrodes[el_id]['NumBytesPerWaveform'] == 0)
                self.__byte_per_waveform_sample[el_id] = self.parameters_electrodes[el_id]['NumBytesPerWaveform']

                # Unused, 10 bytes
                filehandle.read(10)

            elif fileextheader == 'NEUEVLBL':
                # Read electrode number
                # unsigned short 16bit, little endian
                (el_id,) = struct.unpack('<H', filehandle.read(2))

                # Electrode label and remove 0's
                self.parameters_electrodes[el_id]['Label'] = filehandle.read(16).replace('\x00', '')

                # Unused, 6 bytes
                filehandle.read(6)

            elif fileextheader == 'NEUEVFLT':
                filter_types = ('None', 'Butterworth')

                # Read electrode number
                # unsigned short 16bit, little endian
                (el_id,) = struct.unpack('<H', filehandle.read(2))

                # Read high frequency cut-off in mHz
                # unsigned long  32bit, little endian
                (self.parameters_electrodes[el_id]['HiFreqCorner'],) = struct.unpack('<I', filehandle.read(4))

                # Read high frequency filter order
                # unsigned long  32bit, little endian
                (self.parameters_electrodes[el_id]['HiFreqOrder'],) = struct.unpack('<I', filehandle.read(4))

                # Read high frequency filter order
                # unsigned short 16bit, little endian
                (dummy,) = struct.unpack('<H', filehandle.read(2))
                self.parameters_electrodes[el_id]['HiFreqType'] = filter_types[dummy]

                # Read low frequency cut-off in mHz
                # unsigned long  32bit, little endian
                (self.parameters_electrodes[el_id]['LoFreqCorner'],) = struct.unpack('<I', filehandle.read(4))

                # Read low frequency filter order
                # unsigned long  32bit, little endian
                (self.parameters_electrodes[el_id]['LoFreqOrder'],) = struct.unpack('<I', filehandle.read(4))

                # Read low frequency filter order
                # unsigned short 16bit, little endian
                (dummy,) = struct.unpack('<H', filehandle.read(2))
                self.parameters_electrodes[el_id]['LoFreqType'] = filter_types[dummy]

                # Unused, 2 bytes
                filehandle.read(2)

            elif fileextheader == 'DIGLABEL':
                channel_types = ('serial', 'parallel')

                # Read name of digital channel and remove 0's
                self.parameters_nev['DigitalChannelLabel'] = filehandle.read(16).replace('\x00', '')

                # Read mode of channel: 0=serial, 1=parallel
                # signed byte 8bit, little endian
                (dummy,) = struct.unpack('<b', filehandle.read(1))
                self.parameters_nev['DigitalChannelType'] = channel_types[dummy]

                # Unused, 7 bytes
                filehandle.read(7)

            elif fileextheader == 'NSASEXEV':
                digital_enable_types = ('Disabled', 'Enabled')
                analog_enable_types = ('Disabled', 'Enabled on Lo->Hi', 'Enabled on Hi->Lo', 'Enabled on Lo<->Hi')

                # Read frequency of periodic packet generation in ?? (unit), 0 if none
                # unsigned short 16bit, little endian
                (self.parameters_nev['PeriodicPacketGenerator'],) = struct.unpack('<H', filehandle.read(2))

                # Read if digital input triggers events
                # unsigned byte 8bit, little endian
                (dummy,) = struct.unpack('<B', filehandle.read(1))
                self.parameters_nev['DigitialChannelEnable'] = digital_enable_types [dummy & 1]

                for analog_i in range(0, 5):
                    # Read if analog input triggers events
                    # unsigned byte 8bit, little endian
                    (dummy,) = struct.unpack('<B', filehandle.read(1))
                    self.parameters_nev['AnalogEnable'][analog_i] = analog_enable_types [dummy & 3]

                    # Read analog edge detect level in mV
                    # unsigned short 16bit, little endian
                    (self.parameters_nev['AnalogEdgeDetect'][analog_i],) = struct.unpack('<H', filehandle.read(2))

                # Unused, 6 bytes
                filehandle.read(6)

            else:
                self._diagnostic_print("Header ID " + fileextheader + " is not known.")
                filehandle.read(24)


    def _associate(self, sessionname, nsx_override=None, nev_override=None):
        """
        Associates the object with a specified Blackrock session, i.e., a
        combination of a .nsX and .nev file. The meta data is read into the object
        for future reference.

        Args:
            sessionname (string):
                Name of name session to associate with.
            nsx_override (string):
                File name prefix of nsX files (i.e., without the .nsX
                extension). If None, sessionname is used.
                Default: None.
            nev_override (string):
                File name prefix of nev files (i.e., without the .nev
                extension). If None, sessionname is used.
                Default: None.

        Returns:
            -
        """

        # If already associated, disassociate first
        if self.associated:
            raise IOError("Trying to associate an already associated BlackrockIO object.")

        # Create parameter containers
        # Dictionary that holds different parameters read from the .nev file
        self.parameters_nev = {}
        # List of parameter dictionaries for all potential .nsX files 0..9
        self.parameters_nsx = [ {} for _ in range(10) ]
        # a list of dictionaries that holds parameters for each of the different electrodes
        self.parameters_electrodes = []
        # dictionary that holds different parameters about the patient read from the sif file
        self.parameters_patient = {}

        # Save session to be associated to
        self.associated_fileset = sessionname
        self.parameters_nev["SessionName"] = sessionname

        # Set base prefix for nsx and nev files
        if nsx_override == None:
            self.nsx_fileprefix = sessionname
        else:
            self.nsx_fileprefix = nsx_override
        if nev_override == None:
            self.nev_fileprefix = sessionname
        else:
            self.nev_fileprefix = nev_override


        #===============================================================================
        # Scan NSx files
        #===============================================================================

        # List of sampling resolution of different nsX files
        self.analog_res = 10 * [0]

        # List which .nsX files are available, X=0..9
        self.nsx_avail = []

        # List of numbers of recording channels per .nsX file
        self.num_channels_nsx = 10 * [0]

        # Number of packets in each .nsX file
        self.__num_packets_nsx = 10 * [0]

        # Allocate lists of channel IDs
        self.channel_id_nsx = [ [] for _ in range(0, 10) ]

        # .nsX file positions directly after the header, and end of file
        self.__file_nsx_header_end_pos = 10 * [0]
        self.__file_nsx_end_pos = 10 * [0]

        # Test for each .nsX file
        for nsx_i in range(0, 10):
            # Construct file name of .NSx file
            filename_nsx = self.nsx_fileprefix + '.ns' + str(nsx_i)

            # If this .nsX file exists read its header information
            if os.path.exists(filename_nsx):
                self._diagnostic_print("Scanning " + filename_nsx + ".");

                self.nsx_avail.append(nsx_i)

                filehandle = open(filename_nsx, 'rb')

                # Read header information
                self.__read_nsx_header(filehandle, nsx_i)

                # Calculate number of data points and jump to end of headers
                filehandle.seek(0, os.SEEK_END)
                self.__file_nsx_end_pos[nsx_i] = filehandle.tell()
                filehandle.seek(self.__file_nsx_header_end_pos[nsx_i], os.SEEK_SET)

                # Number of data packets per channel in file (each value is 16bit)
                # Subtracting 1 in order not to take the last sample in case it's incomplete
                self.__num_packets_nsx[nsx_i] = (self.__file_nsx_end_pos[nsx_i] - self.__file_nsx_header_end_pos[nsx_i]) / (2 * self.num_channels_nsx[nsx_i]) - 1

                if (self.__file_nsx_end_pos[nsx_i] - self.__file_nsx_header_end_pos[nsx_i]) % (2 * self.num_channels_nsx[nsx_i]) != 0:
                    self._diagnostic_print('Unequal number of packets per channel in file .ns' + str(nsx_i) + '.')

                filehandle.close()

            # Create units for all .nsX
            self.nsx_unit = {}
            for nsx_i in self.nsx_avail:
                self.nsx_unit[nsx_i] = pq.CompoundUnit("1.0/" + str(self.analog_res[nsx_i]) + "*s")

        # Print a warning if no .nsX are found
        if len(self.nsx_avail) == 0:
            self._diagnostic_print("No .nsX files were found for session " + sessionname + ".")


        #===============================================================================
        # Open NEV file
        #===============================================================================

        # Channels in .nev file
        self.channel_id_nev = []

        # Construct file name of .nev file
        filename_nev = self.nev_fileprefix + '.nev'

        # Does the .nev file exist?
        if not os.path.exists(filename_nev):
            self.nev_avail = False
            self._diagnostic_print("No .nev file was found for session " + sessionname + ".")
        else:
            self.nev_avail = True
            self._diagnostic_print("Scanning " + filename_nev + ".")


            #===============================================================================
            # Read headers
            #===============================================================================

            filehandle = open(filename_nev, 'rb')
            # Read basic header block
            self.__read_nev_header(filehandle)
            # Save position after basic file header
            self.__file_nev_header_end_pos = filehandle.tell()
            # Read extended header blocks
            self.__read_nev_ext_header(filehandle)
            # Is the file in a consistent state?
            if not self.__file_nev_ext_header_end_pos == filehandle.tell():
                raise IOError("The .nev file for session " + sessionname + " is corrupt.")


            #===============================================================================
            # Read data packets and read all markers
            #===============================================================================

            # Calculate number of data points from the end of the header to the end of the file
            filehandle.seek(0, os.SEEK_END)
            self.__file_nev_end_pos = filehandle.tell()
            filehandle.seek(self.__file_nev_ext_header_end_pos, os.SEEK_SET)

            # Make sure the number of data bytes is a multiple of the size of data packets
            if not (self.__file_nev_end_pos - self.__file_nev_ext_header_end_pos) % self.__packet_bytes == 0:
                raise IOError("Number of data packets in .nev file for session " + sessionname + " is invalid.")

            # Number of data packets in file (each value is 16bit)
            self.__num_packets_nev = (self.__file_nev_end_pos - self.__file_nev_ext_header_end_pos) / self.__packet_bytes

            # Pre-read buffer for faster input
            filehandle.seek(self.__file_nev_ext_header_end_pos, os.SEEK_SET)
            filebuffer = filehandle.read(self.__packet_bytes * self.__num_packets_nev)

            # Read packets. The byte structure is
            # "<IHBBHhhhhh" + str(self.__packet_bytes - 20) + "s"
            # and the fields are:
            #
            # timestamp: unsigned long 32bit, little endian
            # packet id (0 for experiment event, or electrode id): unsigned short 16bit, little endian
            # classification or reason: unsigned byte 8bit, little endian
            #   packet id =0: insertion reason (bit 0=digital, bits 1-5 analog channels
            #   packet id >0: classifier (0=unclass, 1-16 units 1-16, 255 noise)
            # reserved: unsigned byte 8bit, little endian
            # digital and analog values or waveform
            #   packet id =0: digital input:  unsigned short 16bit, little endian
            #                 analog input 1:   signed short 16bit, little endian
            #                 analog input 2:   signed short 16bit, little endian
            #                 analog input 3:   signed short 16bit, little endian
            #                 analog input 4:   signed short 16bit, little endian
            #                 analog input 5:   signed short 16bit, little endian
            #   packet id >0: waveform: (__packet_bytes-8) unsigned bytes 8bit, little endian
            #
            # All arrays are converted to dtype 'int', which makes subsequent operations considerably faster,
            # at the cost of slightly increased memory usage.
            buf = np.frombuffer(filebuffer, count=-1, dtype='<I').reshape((self.__num_packets_nev, self.__packet_bytes / 4), order='C')
            self._event_timestamps = buf[:, 0].astype('int')

            buf = np.frombuffer(filebuffer, count=-1, dtype='<H').reshape((self.__num_packets_nev, self.__packet_bytes / 2), order='C')
            self._event_packet_id = buf[:, 2].astype('int')
            self._event_digital_marker = buf[:, 4].astype('int')

            buf = np.frombuffer(filebuffer, count=-1, dtype='<h').reshape((self.__num_packets_nev, self.__packet_bytes / 2), order='C')
            self._event_analog_marker = buf[:, 5:10].astype('int')

            buf = np.frombuffer(filebuffer, count=-1, dtype='<B').reshape((self.__num_packets_nev, self.__packet_bytes), order='C')
            self._event_class_or_reason = buf[:, 6].astype('int')


            #===============================================================================
            # Determine the neuron IDs present for each electrode
            #===============================================================================

            # Pre-allocate for each electrode
            self._unit_ids = [ [] for _ in range(256)]

            # Calculate on which channels there are spike events (electrodes in the file are 1-2048 (V<2.3: 1-256))
            self.channel_id_nev = list(np.unique(self._event_packet_id[np.logical_and(self._event_packet_id >= 1, self._event_packet_id <= 2048)]))

            # Determine number of neurons per channel, and sorted IDs
            # Note that this list uses python indexing, i.e., electrode is at list position 0
            for electrode_i in self.channel_id_nev:
                self._unit_ids[electrode_i - 1] = np.unique(self._event_class_or_reason[self._event_packet_id == electrode_i])

            # Calculate indices for all spike events
            (self._spike_index,) = np.nonzero(self._event_packet_id)


            #===============================================================================
            # Determine comment markers
            #===============================================================================

            # Calculate indices for all comments
            (self._comment_index,) = np.nonzero(self._event_packet_id == 0xffff)

            # Save color values of comments
            self._comment_rgba_color = {}
            buf = np.frombuffer(filebuffer, count=-1, dtype='<I').reshape((self.__num_packets_nev, self.__packet_bytes / 4), order='C')
            for marker_idx in self._comment_index:
                self._comment_rgba_color[marker_idx] = buf[marker_idx, 2].astype('int')

            # Save actual comments
            self._comment = {}
            buf = np.frombuffer(filebuffer, count=-1, dtype='<B').reshape((self.__num_packets_nev, self.__packet_bytes), order='C')

            for marker_idx in self._comment_index:
                charset = buf[marker_idx, 6]
                if charset == 0:
                    # ANSI ASCII: Remove unprintable characters (replace by 0 to end string)
                    c = buf[marker_idx, 12:self.__packet_bytes - 12].copy()
                    c[c < 32] = 0
                    c[c > 126] = 0
                    self._comment[marker_idx] = c.tostring()
                elif charset == 1:
                    # UTF-16
                    self._comment[marker_idx] = unicode(buf[marker_idx, 12:self.__packet_bytes - 12].tostring(), encoding='utf16', errors='ignore')


            #===============================================================================
            # Determine the digital marker IDs present in the session
            #===============================================================================

            # For each marker that is digital, record its value
            (self._digital_marker_index,) = np.nonzero(np.logical_and(self._event_packet_id == 0, np.bitwise_and(self._event_class_or_reason, 1)))

            self._digital_marker_ids = np.unique(self._event_digital_marker[self._digital_marker_index])


            #===============================================================================
            # Determine the analog marker IDs present in the session
            #===============================================================================

            # Pre-allocate for 5 analog channels
            self._analog_marker_index = 5 * [ None ]
            self._analog_marker_ids = 5 * [None]

            # Calculate indices for analog marker events
            for analog_i in range(0, 5):
                (self._analog_marker_index[analog_i],) = np.nonzero(np.logical_and(self._event_packet_id == 0, np.bitwise_and(self._event_class_or_reason, 2 ** (analog_i + 1))))
                self._analog_marker_ids[analog_i] = np.unique(self._event_analog_marker[self._analog_marker_index[analog_i], analog_i])

            filehandle.close()

            # Create units for nev
            self.nev_unit = pq.CompoundUnit("1.0/" + str(self.timestamp_res) + "*s")
            self.waveform_unit = pq.CompoundUnit("1.0/" + str(self.waveform_res) + "*s")


        #===============================================================================
        # Finalize association
        #===============================================================================

        # This object is now successfully associated with a session
        self.associated = True


    def get_unit_ids(self, electrode):
        '''
        Returns a list of neuron IDs recorded on a specific electrode. Neuron ID
        0 corresponds to unsorted neurons, ID 255 is the 'noise' channel. IDs
        1-16 are sorted neurons. (cf. Blackrock IO manual)

        Args:
            electrode (int):
                Electrode number 1-255 for which to extract neuron IDs.

        Returns:
            list
                List containing all neuron IDs of the specified electrode.
        '''

        if electrode < 1  or electrode > 2048:
            raise Exception("Invalid electrode ID specified.")

        return self._unit_ids[electrode - 1]


    def get_digital_marker_ids(self):
        '''
        Returns a list of digital marker IDs in the file.

        Args:
            None.

        Returns:
            list
                List containing all digital marker IDs.
        '''

        return self._digital_marker_ids


    def get_analog_marker_ids(self, analog_channel):
        '''
        Returns a list of analog marker IDs in the file.

        Args:
            analog_channel (int):
                Number of the analog channel for which to extract IDs.

        Returns:
            list:
                List containing all analog marker IDs of the specified channel.
        '''

        return self._analog_marker_ids[analog_channel]


    def get_max_time(self):
        '''
        Returns the largest time that can be determined from the recording for
        use as the upper bound m in an interval [n,m).

        Args:
            None.

        Returns:
            Quantity:
                Largest time point (event or analog signal) found in the data,
                either in units of the nev or the nsX file.
                
                Note that this notation is compatible with pythonic indexing: If
                the last sample n of an analog signal is the last sample, or if
                the last time stamp n of spikes/markers is the last sample,
                (n+1)*sample_period is returned. Returns None if no data is
                available.
        '''

        if self.nev_avail:
            tstop = pq.Quantity(np.max(self._event_timestamps + 1), self.nev_unit, dtype=int)
        else:
            tstop = pq.Quantity(0, pq.s, dtype=int)
        for nsx_i in self.nsx_avail:
            last_nsx_time = (self.__num_packets_nsx[nsx_i]) * self.nsx_unit[nsx_i]
            if tstop < last_nsx_time:
                tstop = last_nsx_time

        return tstop


    def read_block(self, lazy=False, cascade=True, n_starts=[None], n_stops=[None], channel_list=[], nsx=[], units=[], events=False, waveforms=False):
        """Reads file contents as a neo Block.

        The Block contains one Segment for each entry in zip(n_starts,
        n_stops). If these parameters are not specified, the default is
        to store all data in one Segment.

        The Block contains one RecordingChannelGroup per channel.

        Args:
            lazy (bool):
                Loads the neo block structure without the following data:
                    signal of AnalogSignal objects 
                    times of SpikeTrain objects
                    channelindexes of RecordingChannelGroup and Unit objects
            cascade (bool):
            n_starts (list):
                List of starting times as Quantity objects of each Segment, where
                the beginning of the file is time 0. If a list entry is specified as
                None, the beginning of the file is used.
                Default: [None].
            n_stops (list):
                List of corresponding stop times as Quantity objects of each
                Segment. If a list entry is specified as None, the end of the
                file is used (including the last sample). Note that the stop
                time follows pythonic indexing conventions: the sample
                corresponding to the stop time is not returned. Default: [None].
            channel_list (list):
                List of channels to consider in loading (Blackrock channel IDs).
                The neural data channels are 1 - 128. The analog inputs are
                129 - 255. If an empty list is specified, all channels are loaded.
                If None is specified, no channels are considered.
                Default: [].
            nsx (list):
                List of integers X (between 0 and 9) specifying the .nsX file to
                load analog data from channels in channel_list from. If an empty
                list is given, all available nsX files are loaded. If None is
                specified, no analog data is read. Blackrock defined sampling
                frequencies:
                    .ns6 - 30000 Hz (no digital filter)
                    .ns5 - 30000 Hz
                    .ns4 - 10000 Hz
                    .ns3 -  2000 Hz
                    .ns2 -  1000 Hz
                    .ns1 -   500 Hz
                Default: [].
            units (list or dictionary):
                Specifies the unit IDs to load from the data. If an empty list is
                specified, all units of all channels in channel_list are loaded. If
                a list of units is given, all units matching one of the IDs in the
                list are loaded from all channels in channel_list. If a dictionary
                is given, each entry with a key N contains a list of unit IDs to
                load from channel N (but only channels in channel_list are loaded).
                If None is specified, no units are loaded.
                Blackrock definition of units:
                        0     unsorted
                        1-16  sorted units
                        255   "noise"
                Default: [].
            events (boolean):
                If True, all digital and analog events are inserted as EventArrays.
                Default: False
            waveforms (boolean):
                If True, waveforms of each spike are loaded in the SpikeTrain
                objects. Default: False

        Returns:
            neo.Block
                neo Block object containing the data.
                Attributes:
                    name: short session name
                    file_origin: session name
                    rec_datetime: date and time
                    description: string "Blackrock format file"

                The neo Block contains the following neo structures:
                Segment
                    For each pair of n_start and n_stop values, one Segment is
                    inserted.
                    Attributes:
                        name: string of the form "Segment i"
                        file_origin: session name
                        rec_datetime: date and time
                        index: consecutive number of the Segment

                RecordingChannelGroup
                    For each recording channel, one RecordingChannelGroup object
                    is created.
                    Attributes:
                        name: string of the form "Channel i"
                        file_origin: session name
                        rec_datetime: date and time
                        channel_indexes: numpy array of one element, which is the integer channel number

                RecordingChannel
                    For each recording channel, one RecordingChannel object is
                    created as child of the respective RecordingChannelGroup.
                    Attributes:
                        name: string of the form "Channel i"
                        file_origin: session name
                        rec_datetime: date and time
                        index: integer channel number

                AnalogSignal
                    For each loaded analog signal, one AnalogSignal object is created per Segment.
                    Attributes:
                        name: string of the form "Analog Signal Segment i, Channel j, NSX x"
                        file_origin: session name
                    Annotations:
                        channel_id (int): channel number
                        file_nsx (int): number X of .nsX file

                Unit
                    For each unit, one Unit structure is created.
                    Attributes:
                        name: string of the form "Channel j, Unit u"
                        file_origin: session name
                        channel_indexes: numpy array of one element, which is the integer channel number
                    Annotations:
                        channel_id (int): channel number
                        unit_id (int): unit number

                SpikeTrain
                    For each Unit and each Segment, one SpikeTrain is created.
                    Waveforms of spikes are inserted into the waveforms property of the
                    respective SpikeTrain objects. Individual Spike objects are not
                    used.
                    Attributes:
                        name: string of the form "Segment i, Channel j, Unit u"
                        file_origin: session name
                        dtype: int (original time stamps are save in units of nev_unit)
                        sampling_rate: Waveform time resolution
                    Annotations:
                        channel_id (int): channel number
                        unit_id (int): unit number

                EventArray
                    For each time-stamped comment in the data file (version >=2.3) one event
                    is created. 
                    Attributes:
                        name: string of the form "Comment n"
                        file_origin: session name
                        dtype: int (original time stamps are save in units of nev_unit)
                        labels: contains a string of the comment number n
                    Annotations:
                        comment (str): comment text
                        comment_rgba_color (int): color of the comment in RGBA format (4 bytes lumped in one integer)

                EventArray
                    Per Segment, for digital or analog markers with a common ID,
                    one EventArray is created containing the time stamps of all
                    occurrences of that specific marker.
                    Attributes:
                        name: string of the form "Digital Marker m" / "Analog Channel i Marker m"
                        file_origin: session name
                        dtype: int (original time stamps are save in units of nev_unit)
                        labels: contains a string of the ID per time stamp
                    Annotations:
                        marker_id (int): marker ID
                        digital_marker (bool): True if it is a digital marker
                        analog_marker (bool):  True if it is an analog marker
                        analog_channel (int): For analog markers, determines the respective channel (0-4)

            Notes:
                For Segment and SpikeTrain objects, if t_start is not specified, it
                is assumed 0. If t_stop is not specified, the maximum time of all
                analog samples available, all available spike time stamps and all
                trigger time stamps is assumed.
        """

        # Make sure this object is associated
        if not self.associated:
            raise IOError("Cannot load from unassociated session.")

        #===============================================================================
        # Input checking and correcting
        #===============================================================================

        # For lazy users that specify x,x instead of [x],[x] for n_starts,n_stops
        if n_starts == None:
            n_starts = [None]
        elif type(n_starts) == pq.Quantity:
            n_starts = [n_starts]
        elif type(n_starts) != list or any([(type(i) != pq.Quantity and i != None) for i in n_starts]):
            print n_starts[0]
            raise ValueError('Invalid specification of n_starts.')
        if n_stops == None:
            n_stops = [None]
        elif type(n_stops) == pq.Quantity:
            n_stops = [n_stops]
        elif type(n_stops) != list or any([(type(i) != pq.Quantity and i != None) for i in n_stops]):
            raise ValueError('Invalid specification of n_stops.')


        # Use all .nsx files?
        if nsx == []:
            nsx = self.nsx_avail

        # If no nsx is specified, convert to an empty list (no nsx files)
        if nsx == None:
            nsx = []

        # Also permit lazy specification for list of nsx
        if type(nsx) == int:
            nsx = [nsx]
        elif type(nsx) != list:
            raise ValueError('Invalid specification of nsx.')


        # Load from all channels?
        if channel_list == []:
            # Determine which channels are available (in all nsx and nev)
            channel_list = []
            for nsx_i in nsx:
                channel_list.extend(self.channel_id_nsx[nsx_i])
            channel_list.extend(self.channel_id_nev)
            channel_list = list(np.unique(channel_list))

        # If no channels are specified, then transform to an empty list
        if channel_list == None:
            channel_list = []

        if type(channel_list) != list:
            raise ValueError('Invalid specification of channel_list.')


        # Make sure the requested .nsX file exists
        for nsx_i in nsx:
            if nsx_i not in self.nsx_avail:
                raise IOError("Data at requested sampling frequency is not found (.ns" + str(nsx_i) + ").")

        # Make sure the requested .nev exists
        if (units != None or events) and not self.nev_avail:
            raise IOError("Requested spiking data and/or events not found (.nev).")


        # Set of all unit IDs that are requested on at least one channel
        complete_unit_ids = set([])

        # Find IDs of all units if complete spike data is requested
        if units == None:
            # Select no units
            units = {}
            for channel_i in channel_list:
                units[channel_i] = []
        elif units == []:
            # Select all units
            units = {}
            for channel_i in channel_list:
                units[channel_i] = self.get_unit_ids(channel_i)
                complete_unit_ids.update(units[channel_i])
        elif type(units) == list or type(units) == int:
            # Select the same unit(s) on all channels
            unit_select = units
            units = {}
            for channel_i in channel_list:
                # Keep only existing units
                units[channel_i] = list(set(self.get_unit_ids(channel_i)) & set(unit_select))
                complete_unit_ids.update(units[channel_i])
        elif type(units) == dict:
            # Fill dictionary with missing entries
            for channel_i in channel_list:
                if not units.has_key(channel_i):
                    units[channel_i] = []
                if type(units[channel_i]) == int:
                    units[channel_i] = [units[channel_i]]
                # Keep only existing units
                units[channel_i] = list(set(self.get_unit_ids(channel_i)) & set(units[channel_i]))
                complete_unit_ids.update(units[channel_i])
        else:
            raise ValueError('Invalid specification of channel_list.')


        if not type(events) == bool:
            raise ValueError('Invalid specification of events.')
        if not type(waveforms) == bool:
            raise ValueError('Invalid specification of waveforms.')


        #=======================================================================
        # Preparations and pre-calculations
        #=======================================================================

        # A list containing the start and stop times (as Quantity) for each segment
        # Used later in creating neo Spiketrain objects
        tstart = []
        tstop = []
        # A list of boolean arrays containing indices into self._event* for
        # each of the segments -- precalculated for speed reasons
        t_idx = [];
        # A dictionary of boolean arrays containing the indices into
        # self._event* for each unit ID -- precalculated for speed reasons
        u_idx = {}

        # Create a neo Block
        if self.nev_avail:
            # Time stamp only available from NEV file
            recdatetime = self.parameters_nev["DateTime"]
        else:
            # Set time stamp to earliest possible date
            recdatetime = datetime(year=datetime.MINYEAR, month=1, day=1)

        bl = neo.Block(name=os.path.basename(self.associated_fileset),
                       description=self.description,
                       file_origin=self.associated_fileset,
                       rec_datetime=recdatetime)

        # Cascade only returns the Block without children, so we are done here
        if not cascade:
            return bl

        # Create a dictionary of segments
        seg = {}
        for (seg_i, n_start_i, n_stop_i) in zip(range(len(n_starts)), n_starts, n_stops):
            # Make sure start time < end time
            if n_start_i != None and n_stop_i != None and n_start_i >= n_stop_i:
                raise ValueError("An n_starts value is larger than the corresponding n_stops value.")

            # If this segments starts at the beginning or lasts until the end,
            # try to find out the total length of recording as good as possible
            # (last event or last sample in analog signal)
            # Also determine start and end packets in nev time stamps
            if n_start_i == None:
                tstart.append(pq.Quantity(0, self.nev_unit, dtype=int))
                start_packet = 0
            else:
                tstart.append(n_start_i)
                start_packet = int(((n_start_i / self.nev_unit).simplified).base)
            if n_stop_i == None:
                tstop.append(self.get_max_time())
                end_packet = max(self._event_timestamps) + 1  # add 1 to get last sample as well (not inclusive)
            else:
                tstop.append(n_stop_i)
                end_packet = int(((n_stop_i / self.nev_unit).simplified).base)

            # Create new neo Segment for this time period
            seg[seg_i] = neo.Segment(name="Segment " + str(seg_i),
                                     file_origin=self.associated_fileset,
                                     rec_datetime=recdatetime,
                                     index=seg_i,
                                     t_start=tstart[-1],
                                     t_stop=tstop[-1])
            bl.segments.append(seg[seg_i])
            # TODO: ref to parent required by spyke viewer (see TODO above)?
            seg[seg_i].block = bl

            # Pre-calculate the indices of nev time stamps corresponding to the segment
            # Obey pythonic indexing, e.g., sample end_packet is not considered!
            t_idx.append((self._event_timestamps >= start_packet) & (self._event_timestamps < end_packet))

        # Create a dictionary of recording channels, recordings channel groups, and units
        rc = {}
        rcg = {}
        un = dict([ [i, {}] for i in channel_list])
        for channel_i in channel_list:
            rc[channel_i] = neo.RecordingChannel(name="Channel " + str(channel_i),
                                                 index=channel_i,
                                                 file_origin=self.associated_fileset,
                                                 rec_datetime=recdatetime)

            if not lazy:
                data = np.array([channel_i], dtype=int)
            else:
                data = np.array([], dtype=int)
            rcg[channel_i] = neo.RecordingChannelGroup(name="Channel " + str(channel_i),
                                                       channel_indexes=data,
                                                       file_origin=self.associated_fileset,
                                                       rec_datetime=recdatetime)
            if lazy:
                rcg[channel_i].lazy_shape = True

            for unit_i in units[channel_i]:
                un[channel_i][unit_i] = neo.Unit(channel_indexes=data,
                                                 name="Channel " + str(channel_i) + ", Unit " + str(unit_i),
                                                 file_origin=self.associated_fileset,
                                                 channel_id=channel_i,
                                                 unit_id=unit_i)
                if lazy:
                    un[channel_i][unit_i].lazy_shape = True

                rcg[channel_i].units.append(un[channel_i][unit_i])
                un[channel_i][unit_i].recordingchannelgroup = rcg[channel_i]

            # TODO: Is this reference from RCG to the BLOCK a neo specification
            # Seems to be required by spyke viewer?
            rcg[channel_i].block = bl
            rcg[channel_i].recordingchannels.append(rc[channel_i])

            rc[channel_i].recordingchannelgroups.append(rcg[channel_i])

            bl.recordingchannelgroups.append(rcg[channel_i])

        # Precalculate indices of those unit IDs that are going to be loaded for speed-up
        for unit_i in complete_unit_ids:
            u_idx[unit_i] = self._event_class_or_reason == unit_i

        # If spike waveforms are requested, pre-read the nev file packets for speed reasons
        if waveforms and not lazy:
            # Construct file name of .nev file
            filename_nev = self.nev_fileprefix + '.nev'
            # The .nev must still exist
            if not os.path.isfile(filename_nev):
                raise IOError(".nev file for session " + self.associated_fileset + " not found.")

            try:
                # Open the .nev file
                filehandle_nev = open(filename_nev, 'rb')

                # Pre-read buffer for faster input
                filehandle_nev.seek(self.__file_nev_ext_header_end_pos, os.SEEK_SET)
                filebuffer = filehandle_nev.read(self.__packet_bytes * self.__num_packets_nev)
            finally:
                filehandle_nev.close()


        #=======================================================================
        # Read neo Block
        #=======================================================================

        #----------------------------------------------------- Load nev data

        # To avoid unnecessary conversions when all channels have waveforms
        # sampled at the same bandwidth
        last_numbytes = 0

        # Go through all requested channels
        for channel_i in channel_list:
            # Determine corresponding channel indices
            ch_idx = (self._event_packet_id == channel_i)

            # Load all waveforms of that channel if required
            if waveforms:
                # Determine digital resolution of waveforms, and allocate array accordingly
                datatypes = ["", "<b", "<h", "", "<i"]
                numbytes = self.__byte_per_waveform_sample[channel_i]

                # Pre-allocate waveform array: one row per spike, one column per waveform sample
                if numbytes not in [1, 2, 4]:
                    raise IOError("Invalid number of bytes per waveform sample (" + str(numbytes) + ").")

                # Only read data again if data type is different from last channel
                if last_numbytes == 0 or numbytes != last_numbytes:
                    # Read all packets at once for speed reasons and reshape to matrix of that channel
                    #  timestamp; unsigned long 32bit, little endian
                    #  packet id (0 for experiment event, or electrode id): unsigned short 16bit, little endian
                    #  classification or reason: unsigned byte 8bit, little endian
                    #    packet id >0: classifier (0=unclass, 1-16 units 1-16, 255 noise)
                    #  reserved: unsigned byte 8bit, little endian
                    #  waveform: (__packet_bytes-8) unsigned bytes 8bit, little endian
                    # wfbuf_reduced = np.frombuffer(filebuffer, count=-1, dtype=datatypes[numbytes]).reshape((-1, self.__packet_bytes / numbytes), order='C')[ch_idx, (8 / numbytes):self.__packet_bytes / numbytes]
                    wfbuf = np.frombuffer(filebuffer, count=-1, dtype=datatypes[numbytes]).reshape((-1, self.__packet_bytes / numbytes), order='C')[:, (8 / numbytes):self.__packet_bytes / numbytes]
                    last_numbytes = numbytes

            for (seg_i, n_start_i, n_stop_i) in zip(range(len(n_starts)), n_starts, n_stops):
                for unit_i in units[channel_i]:
                    # Extract all time stamps of that neuron on that electrode
                    combi_idx = ch_idx & t_idx[seg_i] & u_idx[unit_i]

                    if not lazy:
                        data = pq.Quantity(self._event_timestamps[combi_idx], units=self.nev_unit)
                    else:
                        data = pq.Quantity([], units=self.nev_unit)

                    # Create SpikeTrain object
                    st = neo.SpikeTrain(times=data,
                                        dtype='int',
                                        t_start=tstart[seg_i],
                                        t_stop=tstop[seg_i],
                                        sampling_rate=self.nev_unit,
                                        name="Segment " + str(seg_i) + ", Channel " + str(channel_i) + ", Unit " + str(unit_i),
                                        file_origin=self.associated_fileset,
                                        unit_id=unit_i,
                                        channel_id=channel_i)

                    if lazy:
                        st.lazy_shape = True

                    # Attach spike train and unit to neo object
                    un[channel_i][unit_i].spiketrains.append(st)
                    st.unit = un[channel_i][unit_i]
                    seg[seg_i].spiketrains.append(st)
                    st.segment = seg[seg_i]

                    if waveforms and len(combi_idx) > 0 and not lazy:
                        # Collect all waveforms of the specific unit
                        # For computational reasons: no units, no time axis
                        st.waveforms = wfbuf[combi_idx, :].copy()

        if events:
            for (seg_i, n_start_i, n_stop_i) in zip(range(len(n_starts)), n_starts, n_stops):
                for marker_i in self._digital_marker_ids:
                    # Extract all time stamps of digital markers
                    marker_idx = self._digital_marker_index[np.logical_and(self._event_digital_marker[self._digital_marker_index] == marker_i, t_idx[seg_i][self._digital_marker_index])]

                    ev = neo.EventArray(times=pq.Quantity(self._event_timestamps[marker_idx], units=self.nev_unit, dtype="int"),
                                        labels=np.tile(str(marker_i), (len(marker_idx))),
                                        name="Digital Marker " + str(marker_i),
                                        file_origin=self.associated_fileset,
                                        marker_id=marker_i,
                                        digital_marker=True,
                                        analog_marker=False,
                                        analog_channel=0)
                    seg[seg_i].eventarrays.append(ev)

                for analog_channel_i in xrange(5):
                    for marker_i in self._analog_marker_ids[analog_channel_i]:
                        # Extract all time stamps of analog markers in channel analog_channel_i
                        marker_idx = self._analog_marker_index[analog_channel_i][np.logical_and(self._event_analog_marker[analog_channel_i][self._analog_marker_index[analog_channel_i]] == marker_i, t_idx[seg_i][self._analog_marker_index[analog_channel_i]])]

                        ev = neo.EventArray(times=pq.Quantity(self._event_timestamps[marker_idx], units=self.nev_unit, dtype="int"),
                                            name="Analog Channel " + str(analog_channel_i) + ", Marker " + str(marker_i),
                                            labels=np.tile(str(marker_i), (len(marker_idx))),
                                            marker_id=marker_i,
                                            digital_marker=False,
                                            analog_marker=True,
                                            analog_channel=analog_channel_i)
                        seg[seg_i].eventarrays.append(ev)

                # Add time marked comments
                for marker_i, marker_idx in enumerate(self._comment_index):
                    if t_idx[seg_i][marker_idx]:
                        ev = neo.EventArray(times=pq.Quantity(self._event_timestamps[marker_idx], units=self.nev_unit, dtype="int"),
                                            name="Comment " + str(marker_i),
                                            labels=str(marker_i),
                                            comment=self._comment[marker_idx],
                                            comment_rgba_color=self._comment_rgba_color[marker_idx])
                        seg[seg_i].events.append(ev)


        #----------------------------------------------------- Load nsX data

        for nsx_i in nsx:
            # Construct the filename of the .nsX file
            filename_nsx = self.nsx_fileprefix + '.ns' + str(nsx_i)

            # The .nsX must still exist
            if not os.path.isfile(filename_nsx):
                raise IOError(".ns" + str(nsx_i) + " file for session " + self.associated_fileset + " not found.")


            # Determine position where analog data starts
            fileoffset = self.__file_nsx_header_end_pos[nsx_i]

            # From version 2.2+, there is a header block to consider
            if float(self.parameters_nsx[nsx_i]['Version']) > 2.1:
                try:
                    # Open the .nev file
                    filehandle_nsx = open(filename_nsx, 'rb')

                    # Read all header data bytes
                    filehandle_nsx.seek(fileoffset , os.SEEK_SET)
                    temp1 = np.fromfile(filehandle_nsx, count=1, dtype='b')  # should be 1
                    temp2 = np.fromfile(filehandle_nsx, count=2, dtype='i')  # number of data points that follow (per channel!)
                    #TODO: temp2[0] is the time stamp of the first sample -> should be recognized!!!
                    if temp1[0] != 1 or temp2[1] != (self.__num_packets_nsx[nsx_i] + 1):
                        raise Exception('blackrockio cannot handle files with gaps (available in version 2.2+).')

                    # For 2.3 files, check that only one header block follows (i.e., no gaps)
                    # or, even better, deal with having more than one block of LFP data!
                    # TODO: Implement this functionality instead of raising an exception
                finally:
                    filehandle_nsx.close()
                    # add the 9 bytes (b+2*i) to fileoffset to
                    fileoffset = fileoffset + 9

            # Explicitely mention the number of bytes to read from the
            # file using count=... because we disregard the last sample
            # point (see definition of self.__num_packets_nsx above)
            analogbuf = np.memmap(filename_nsx, dtype='<h', mode='r', offset=fileoffset, shape=(self.__num_packets_nsx[nsx_i], self.num_channels_nsx[nsx_i]))

            # Go through all time periods
            for (seg_i, n_start_i, n_stop_i) in zip(range(len(n_starts)), n_starts, n_stops):
                # Start and end packet to read
                if n_start_i != None:
                    start_packet = int(((n_start_i / self.nsx_unit[nsx_i]).simplified).base)
                    if start_packet < 0:
                        start_packet = 0
                    if start_packet > self.__num_packets_nsx[nsx_i]:
                        start_packet = self.__num_packets_nsx[nsx_i]
                else:
                    start_packet = 0

                if n_stop_i != None:
                    end_packet = int(((n_stop_i / self.nsx_unit[nsx_i]).simplified).base)
                    if end_packet < 0:
                        end_packet = 0
                    if end_packet > self.__num_packets_nsx[nsx_i] :
                        end_packet = self.__num_packets_nsx[nsx_i]
                else:
                    end_packet = self.__num_packets_nsx[nsx_i]

                # Calculate the sequential signal number in the file
                el_idx = [self.channel_id_nsx[nsx_i].index(i) for i in channel_list if i in self.channel_id_nsx[nsx_i]]

                # Load individual channels
                for (channel_i, el_idx_i) in zip(channel_list, el_idx):
                    # Create a unit for the amplitude of the channel voltage
                    try:
                        digif = self.parameters_electrodes[channel_i]['DigitizationFactor']
                        if digif == 1:
                            LFPunit = pq.CompoundUnit('10^-9*V')
                        elif digif == 1000:
                            LFPunit = pq.CompoundUnit('10^-6*V')
                        elif digif == 1000000:
                            LFPunit = pq.mV
                        elif digif == 1000000000:
                            LFPunit = pq.V
                        else:
                            # Other more complicated gain factor
                            LFPunit = pq.CompoundUnit(str(digif) + '*10^-9*V')
                    except:
                        # If no nev file exists, or the information is missing for other reasons,
                        # return channel data without unit attached (i.e., in voltage steps)
                        LFPunit = pq.dimensionless
                        self._diagnostic_print("Warning: Channel " + str(channel_i) + " does not have a digitization factor -- data is dimensionless.")

                    if not lazy:
                        data = pq.Quantity(analogbuf[start_packet:end_packet, el_idx_i].T, units=LFPunit)
                    else:
                        data = pq.Quantity([], units=LFPunit)

                    asig = neo.AnalogSignal(signal=data,
                                            sampling_period=self.nsx_unit[nsx_i],
                                            # Alternative:
                                            # sampling_rate=pq.CompoundUnit(str(self.analog_res[nsx_i]) + ' * Hz'),
                                            t_start=start_packet * self.nsx_unit[nsx_i],
                                            name="Analog Signal Segment " + str(seg_i) + ", Channel " + str(channel_i) + ", NSX " + str(nsx_i),
                                            file_origin=self.associated_fileset,
                                            channel_id=channel_i,
                                            file_nsx=nsx_i)

                    if lazy:
                        asig.lazy_shape = True


                    # Attach analog signal to segment and recording channel
                    seg[seg_i].analogsignals.append(asig)
                    rc[channel_i].analogsignals.append(asig)
                    asig.segment = seg[seg_i]
                    asig.recordingchannel = rc[channel_i]

        return bl


    def __str__(self):
        print self.associated_fileset
        print " "
        if self.nev_avail:
            print(" ")
            print("Event Parameters (NEV)")
            print("====================================")
            print "Timestamp resolution (Hz): ", str(self.timestamp_res)
            print "Waveform resolution (Hz): ", str(self.timestamp_res)
            print "Available electrode IDs: ", str(self.channel_id_nev)
            for key_i in self.parameters_nev.keys():
                print(key_i + ": " + str(self.parameters_nev[key_i]))
        for nsx_i in self.nsx_avail:
            print(" ")
            print("Analog Parameters (NS" + str(nsx_i) + ")")
            print("====================================")
            print "Resolution (Hz): ", self.analog_res[nsx_i]
            print "Available channel IDs: ", self.channel_id_nsx[nsx_i]
            for key_i in self.parameters_nsx[nsx_i].keys():
                print(key_i + ": " + str(self.parameters_nsx[nsx_i][key_i]))
