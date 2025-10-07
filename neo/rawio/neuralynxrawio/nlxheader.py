from packaging.version import Version
import os
import re
from collections import OrderedDict

from neo.rawio.neuralynxrawio.ncssections import AcqType


class NlxHeader(OrderedDict):
    """
    Representation of basic information in all 16 kbytes Neuralynx file headers,
    including dates opened and closed if given.

    The OrderedDict contains entries for each property given in the header with '-' in front
    of the key value as well as an 'ApplicationName', 'ApplicationVersion', 'recording_opened'
    and 'recording_closed' entries. The 'InputRange', 'channel_ids', 'channel_names' and
    'bit_to_microvolt' properties are set to lists of entries for each channel which may be
    in the file.
    """

    HEADER_SIZE = 2**14  # Neuralynx files have a txt header of 16kB

    # helper function to interpret boolean keys
    def _to_bool(txt):
        if txt == "True":
            return True
        elif txt == "False":
            return False
        else:
            raise Exception("Can not convert %s to bool" % txt)

    # Keys that may be present in header which we parse. First entry of tuple is what is
    # present in header, second entry is key which will be used in dictionary, third entry
    # type the value will be converted to.
    txt_header_keys = [
        ("AcqEntName", "channel_names", None),  # used
        ("FileType", "", None),
        ("FileVersion", "", None),
        ("RecordSize", "", None),
        ("HardwareSubSystemName", "", None),
        ("HardwareSubSystemType", "", None),
        ("SamplingFrequency", "sampling_rate", float),  # used
        ("ADMaxValue", "", None),
        ("ADBitVolts", "bit_to_microVolt", None),  # used
        ("NumADChannels", "", None),
        ("ADChannel", "channel_ids", None),  # used
        ("InputRange", "", None),
        ("InputInverted", "input_inverted", _to_bool),  # used
        ("DSPLowCutFilterEnabled", "", None),
        ("DspLowCutFrequency", "", None),
        ("DspLowCutNumTaps", "", None),
        ("DspLowCutFilterType", "", None),
        ("DSPHighCutFilterEnabled", "", None),
        ("DspHighCutFrequency", "", None),
        ("DspHighCutNumTaps", "", None),
        ("DspHighCutFilterType", "", None),
        ("DspDelayCompensation", "", None),
        # DspFilterDelay key with flexible µ symbol matching
        # Different Neuralynx versions encode the µ (micro) symbol differently:
        # - Some files use single-byte encoding (latin-1): DspFilterDelay_µs (raw bytes: \xb5)
        # - Other files use UTF-8 encoding: DspFilterDelay_µs (raw bytes: \xc2\xb5)
        # When UTF-8 encoded µ (\xc2\xb5) is decoded with latin-1, it becomes "Âµ"
        # This regex matches both variants: "µs" and "Âµs" but normalizes to "DspFilterDelay_µs"
        (r"DspFilterDelay_[Â]?µs", "DspFilterDelay_µs", None),
        ("DisabledSubChannels", "", None),
        ("WaveformLength", "", int),
        ("AlignmentPt", "", None),
        ("ThreshVal", "", None),
        ("MinRetriggerSamples", "", None),
        ("SpikeRetriggerTime", "", None),
        ("DualThresholding", "", None),
        (r"Feature \w+ \d+", "", None),
        ("SessionUUID", "", None),
        ("FileUUID", "", None),
        ("CheetahRev", "", None),  # only for older versions of Cheetah
        ("ProbeName", "", None),
        ("OriginalFileName", "", None),
        ("TimeCreated", "", None),
        ("TimeClosed", "", None),
        ("ApplicationName", "", None),  # also include version number when present
        ("AcquisitionSystem", "", None),
        ("ReferenceChannel", "", None),
        ("NLX_Base_Class_Type", "", None),  # in version 4 and earlier versions of Cheetah
    ]

    # Filename and datetime may appear in header lines starting with # at
    # beginning of header or in later versions as a property. The exact format
    # used depends on the application name and its version as well as the
    # -FileVersion property. Examples of each known case are shown below in comments.
    #
    # There are now separate patterns for the in header line and in properties cases which cover
    # the known variations in each case. They are compiled here once for efficiency.
    _openDatetime1_pat = re.compile(
        r"## (Time|Date) Opened:* \((m/d/y|mm/dd/yyy)\): (?P<date>\S+)" r"\s+(\(h:m:s\.ms\)|At Time:) (?P<time>\S+)"
    )
    _openDatetime2_pat = re.compile(r"-TimeCreated (?P<date>\S+) (?P<time>\S+)")
    _closeDatetime1_pat = re.compile(
        r"## (Time|Date) Closed:* \((m/d/y|mm/dd/yyy)\): " r"(?P<date>\S+)\s+(\(h:m:s\.ms\)|At Time:) (?P<time>\S+)"
    )
    _closeDatetime2_pat = re.compile(r"-TimeClosed (?P<date>\S+) (?P<time>\S+)")

    # Precompiled filename pattern
    _filename_pat = re.compile(r"## File Name:* (?P<filename>.*)")

    # BML - example
    # ######## Neuralynx Data File Header
    # ## File Name: null
    # ## Time Opened: (m/d/y): 12/11/15  At Time: 11:37:39.000

    # Cheetah after version 1 and before version 5 - example
    # ######## Neuralynx Data File Header
    # ## File Name F:\2000-01-01_18-28-39\RMH3.ncs
    # ## Time Opened (m/d/y): 1/1/2000  (h:m:s.ms) 18:28:39.821
    # ## Time Closed (m/d/y): 1/1/2000  (h:m:s.ms) 9:58:41.322

    # Cheetah version 4.0.2 - example
    # ######## Neuralynx Data File Header
    # ## File Name: D:\Cheetah_Data\2003-10-4_10-2-58\CSC14.Ncs
    # ## Time Opened: (m/d/y): 10/4/2003  At Time: 10:3:0.578

    # Cheetah version 5.4.0 - example
    # ######## Neuralynx Data File Header
    # ## File Name C:\CheetahData\2000-01-01_00-00-00\CSC5.ncs
    # ## Time Opened (m/d/y): 1/01/2001  At Time: 0:00:00.000
    # ## Time Closed (m/d/y): 1/01/2001  At Time: 00:00:00.000

    # Cheetah version 5.5.1 - example
    # ######## Neuralynx Data File Header
    # ## File Name C:\CheetahData\2013-11-29_17-05-05\Tet3a.ncs
    # ## Time Opened (m/d/y): 11/29/2013  (h:m:s.ms) 17:5:16.793
    # ## Time Closed (m/d/y): 11/29/2013  (h:m:s.ms) 18:3:13.603

    # Cheetah version 5.6.0 - example
    # ## File Name: F:\processing\sum-big-board\252-1375\recording-20180107\2018-01-07_15-14-54\04. tmaze1-no-light-start To tmaze1-light-stop\VT1_fixed.nvt
    # ## Time Opened: (m/d/y): 2/5/2018 At Time: 18:5:12.654

    # Cheetah version 5.6.3 - example
    # ######## Neuralynx Data File Header
    # ## File Name C:\CheetahData\2016-11-28_21-50-00\CSC1.ncs
    # ## Time Opened (m/d/y): 11/28/2016  (h:m:s.ms) 21:50:33.322
    # ## Time Closed (m/d/y): 11/28/2016  (h:m:s.ms) 22:44:41.145

    # Cheetah version 5.7.4 - example
    # ######## Neuralynx Data File Header
    # and then properties
    # -OriginalFileName "C:\CheetahData\2017-02-16_17-55-55\CSC1.ncs"
    # -TimeCreated 2017/02/16 17:56:04
    # -TimeClosed 2017/02/16 18:01:18

    # Cheetah version 6.3.2 - example
    # ######## Neuralynx Data File Header
    # and then properties
    # -OriginalFileName "G:\CheetahDataD\2019-07-12_13-21-32\CSC1.ncs"
    # -TimeCreated 2019/07/12 13:21:32
    # -TimeClosed 2019/07/12 15:07:55

    # Cheetah version 6.4.1dev - example
    # ######## Neuralynx Data File Header
    # and then properties
    # -OriginalFileName "D:\CheetahData\2021-02-26_15-46-33\CSC1.ncs"
    # -TimeCreated 2021/02/26 15:46:52
    # -TimeClosed 2021/10/12 09:07:58

    # neuraview version 2 - example
    # ######## Neuralynx Data File Header
    # ## File Name: L:\McHugh Lab\Recording\2015-06-24_18-05-11\NeuraviewEventMarkers-20151214_SleepScore.nev
    # ## Date Opened: (mm/dd/yyy): 12/14/2015 At Time: 15:58:32
    # ## Date Closed: (mm/dd/yyy): 12/14/2015 At Time: 15:58:32

    # pegasus version 2.1.1 and Cheetah beyond version 5.6.4 - example
    # ######## Neuralynx Data File Header
    # and then properties
    # -OriginalFileName D:\Pegasus Data\Dr_NM\1902\2019-06-28_17-36-50\Events_0008.nev
    # -TimeCreated 2019/06/28 17:36:50
    # -TimeClosed 2019/06/28 17:45:48

    def __init__(self, filename, props_only=False):
        """
        Factory function to build NlxHeader for a given file.

        :param filename: name of Neuralynx file
        :param props_only: if true, will not try and read time and date or check start
        """
        super(OrderedDict, self).__init__()

        txt_header = NlxHeader.get_text_header(filename)

        # must start with 8 # characters
        if not props_only and not txt_header.startswith("########"):
            ValueError("Neuralynx files must start with 8 # characters.")

        self._readProperties(filename, txt_header)
        self._setApplicationAndVersion()
        numChidEntries = self._convertChannelIdsNames(filename)
        self._setBitToMicroVolt()
        self._setInputRanges(numChidEntries)
        self._setFilenameProp(txt_header)

        if not props_only:
            self._setTimeDate(txt_header)

        # Normalize all types to proper Python types
        self._normalize_types()

    @staticmethod
    def get_text_header(filename):
        """
        Accessory method to extract text in header. Useful for subclasses.
        :param filename: name of Neuralynx file
        """
        with open(filename, "rb") as f:
            txt_header = f.read(NlxHeader.HEADER_SIZE)
        return txt_header.strip(b"\x00").decode("latin-1")

    def _readProperties(self, filename, txt_header):
        """
        Read properties from header and place in OrderedDictionary which this object is.
        :param filename: name of ncs file, used for extracting channel number
        :param txt_header: header text
        """
        # find keys
        for k1, k2, type_ in NlxHeader.txt_header_keys:
            pattern = r"-(?P<name>" + k1 + r")\s+(?P<value>[\S ]*)"
            matches = re.findall(pattern, txt_header)
            for match in matches:
                if k2 == "":
                    name = match[0]
                else:
                    name = k2
                value = match[1].rstrip(" ")
                if type_ is not None:
                    value = type_(value)
                self[name] = value

    def _setApplicationAndVersion(self):
        """
        Set "ApplicationName" property and app_version attribute based on existing properties
        """
        # older Cheetah versions with CheetahRev property
        if "CheetahRev" in self:
            assert "ApplicationName" not in self
            self["ApplicationName"] = "Cheetah"
            app_version = self["CheetahRev"]
        # new file version 3.4 does not contain CheetahRev property, but ApplicationName instead
        elif "ApplicationName" in self:
            pattern = r'(\S*) "([\S ]*)"'
            match = re.findall(pattern, self["ApplicationName"])
            assert len(match) == 1, "impossible to find application name and version"
            self["ApplicationName"], app_version = match[0]
        # BML Ncs file contain neither property, but 'NLX_Base_Class_Type'
        elif "NLX_Base_Class_Type" in self:
            self["ApplicationName"] = "BML"
            app_version = "1.0"
        # Neuraview Ncs file contained neither property nor NLX_Base_Class_Type information
        else:
            self["ApplicationName"] = "Neuraview"
            app_version = "2"

        if " Development" in app_version:
            app_version = app_version.replace(" Development", ".dev0")

        self["ApplicationVersion"] = Version(app_version)

    def _convertChannelIdsNames(self, filename):
        """
        Convert channel ids and channel name properties, if present.

        :return number of channel id entries
        """
        # if channel_ids or names not in self then the filename is used for channel name
        name = os.path.splitext(os.path.basename(filename))[0]

        # convert channel ids
        if "channel_ids" in self:
            chid_entries = re.findall(r"\S+", self["channel_ids"])
            self["channel_ids"] = [int(c) for c in chid_entries]
        else:
            self["channel_ids"] = ["unknown"]
            chid_entries = []

        # convert channel names
        if "channel_names" in self:
            name_entries = re.findall(r"\S+", self["channel_names"])
            if len(name_entries) == 1:
                self["channel_names"] = name_entries * len(self["channel_ids"])
            assert len(self["channel_names"]) == len(
                self["channel_ids"]
            ), "Number of channel ids does not match channel names."
        else:
            self["channel_names"] = ["unknown"] * len(self["channel_ids"])

        return len(chid_entries)

    def _setBitToMicroVolt(self):
        # convert bit_to_microvolt
        if "bit_to_microVolt" in self:
            btm_entries = re.findall(r"\S+", self["bit_to_microVolt"])
            if len(btm_entries) == 1:
                btm_entries = btm_entries * len(self["channel_ids"])
            self["bit_to_microVolt"] = [float(e) * 1e6 for e in btm_entries]
            assert len(self["bit_to_microVolt"]) == len(
                self["channel_ids"]
            ), "Number of channel ids does not match bit_to_microVolt conversion factors."

    def _setInputRanges(self, numChidEntries):
        if "InputRange" in self:
            ir_entries = re.findall(r"\w+", self["InputRange"])
            if len(ir_entries) == 1:
                self["InputRange"] = [int(ir_entries[0])] * numChidEntries
            else:
                self["InputRange"] = [int(e) for e in ir_entries]
            assert len(self["InputRange"]) == numChidEntries, "Number of channel ids does not match input range values."

    def _setFilenameProp(self, txt_header):
        """
        Add an OriginalFileName property if in header.
        """
        if not "OriginalFileName" in self.keys():
            fnm = NlxHeader._filename_pat.search(txt_header)
            if not fnm:
                return
            else:
                self["OriginalFileName"] = fnm.group(1).strip(' "\t\r\n')
        else:
            # some file types quote the property so strip that also
            self["OriginalFileName"] = self["OriginalFileName"].strip(' "\t\r\n')

    def _setTimeDate(self, txt_header):
        """
        Read time and date from text of header
        """
        import dateutil

        # opening time
        sr = NlxHeader._openDatetime1_pat.search(txt_header)
        if not sr:
            sr = NlxHeader._openDatetime2_pat.search(txt_header)
        if not sr:
            raise IOError(
                f"No matching header open date/time for application {self['ApplicationName']} "
                + f"version {self['ApplicationVersion']}. Please contact developers."
            )
        else:
            dt1 = sr.groupdict()
            self["recording_opened"] = dateutil.parser.parse(f"{dt1['date']} {dt1['time']}")

        # close time, if available
        sr = NlxHeader._closeDatetime1_pat.search(txt_header)
        if not sr:
            sr = NlxHeader._closeDatetime2_pat.search(txt_header)
        if sr:
            dt2 = sr.groupdict()
            self["recording_closed"] = dateutil.parser.parse(f"{dt2['date']} {dt2['time']}")

    def _normalize_types(self):
        """
        Convert all header values to proper Python types.

        This ensures that:
        - Boolean strings ('True', 'False', 'Enabled', 'Disabled') become Python bools
        - Numeric strings ('0.1', '8000') become Python floats/ints
        - Single-element lists are extracted to scalars (for single-channel files)

        This normalization makes the header values directly usable for
        stream identification without additional conversion in NeuralynxRawIO.
        """

        # Convert boolean strings to actual booleans
        bool_keys = [
            'DSPLowCutFilterEnabled',
            'DSPHighCutFilterEnabled',
            'DspDelayCompensation',
        ]

        for key in bool_keys:
            if key in self and isinstance(self[key], str):
                if self[key] in ('True', 'Enabled'):
                    self[key] = True
                elif self[key] in ('False', 'Disabled'):
                    self[key] = False

        # Convert numeric strings to numbers
        numeric_keys = [
            'DspLowCutFrequency',
            'DspHighCutFrequency',
            'DspLowCutNumTaps',
            'DspHighCutNumTaps',
        ]

        for key in numeric_keys:
            if key in self and isinstance(self[key], str):
                try:
                    # Try int first
                    if '.' not in self[key]:
                        self[key] = int(self[key])
                    else:
                        self[key] = float(self[key])
                except ValueError:
                    # Keep as string if conversion fails
                    pass

        # Handle DspFilterDelay_µs (could be string or already converted)
        delay_key = 'DspFilterDelay_µs'
        if delay_key in self and isinstance(self[delay_key], str):
            try:
                self[delay_key] = int(self[delay_key])
            except ValueError:
                pass

        # Extract single-channel InputRange from list
        # For multi-channel files, keep as list
        # For single-channel files, extract the single value
        if 'InputRange' in self and isinstance(self['InputRange'], list):
            if len(self['InputRange']) == 1:
                # Single channel file: extract the value
                self['InputRange'] = self['InputRange'][0]
            # else: multi-channel, keep as list

    def type_of_recording(self):
        """
        Determines type of recording in Ncs file with this header.

        RETURN: NcsSections.AcqType
        """

        if "NLX_Base_Class_Type" in self:

            # older style standard neuralynx acquisition with rounded sampling frequency
            if self["NLX_Base_Class_Type"] == "CscAcqEnt":
                return AcqType.PRE4

            # BML style with fractional frequency and microsPerSamp
            elif self["NLX_Base_Class_Type"] == "BmlAcq":
                return AcqType.BML

            else:
                return AcqType.UNKNOWN

        elif "HardwareSubSystemType" in self:

            # DigitalLynx
            if self["HardwareSubSystemType"] == "DigitalLynx":
                return AcqType.DIGITALLYNX

            # DigitalLynxSX
            elif self["HardwareSubSystemType"] == "DigitalLynxSX":
                return AcqType.DIGITALLYNXSX

            # Cheetah64
            elif self["HardwareSubSystemType"] == "Cheetah64":
                return AcqType.CHEETAH64

            # RawDataFile
            elif self["HardwareSubSystemType"] == "RawDataFile":
                return AcqType.RAWDATAFILE

            else:
                return AcqType.UNKNOWN

        elif "FileType" in self:

            if "FileVersion" in self and self["FileVersion"] in ["3.2", "3.3", "3.4"]:
                return AcqType[self["AcquisitionSystem"].split()[1].upper()]

            else:
                return AcqType.CHEETAH560  # only known case of FileType without FileVersion

        else:
            return AcqType.UNKNOWN
