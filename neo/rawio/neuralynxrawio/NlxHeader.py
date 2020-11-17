import datetime
import distutils.version
import os
import re
from collections import OrderedDict


class NlxHeader(OrderedDict):
    """
    Representation of basic information in all 16 kbytes Neuralynx file headers,
    including dates opened and closed if given.
    """

    HEADER_SIZE = 2 ** 14  # Neuralynx files have a txt header of 16kB

    # helper function to interpret boolean keys
    def _to_bool(txt):
        if txt == 'True':
            return True
        elif txt == 'False':
            return False
        else:
            raise Exception('Can not convert %s to bool' % txt)

    # keys that may be present in header which we parse
    txt_header_keys = [
        ('AcqEntName', 'channel_names', None),  # used
        ('FileType', '', None),
        ('FileVersion', '', None),
        ('RecordSize', '', None),
        ('HardwareSubSystemName', '', None),
        ('HardwareSubSystemType', '', None),
        ('SamplingFrequency', 'sampling_rate', float),  # used
        ('ADMaxValue', '', None),
        ('ADBitVolts', 'bit_to_microVolt', None),  # used
        ('NumADChannels', '', None),
        ('ADChannel', 'channel_ids', None),  # used
        ('InputRange', '', None),
        ('InputInverted', 'input_inverted', _to_bool),  # used
        ('DSPLowCutFilterEnabled', '', None),
        ('DspLowCutFrequency', '', None),
        ('DspLowCutNumTaps', '', None),
        ('DspLowCutFilterType', '', None),
        ('DSPHighCutFilterEnabled', '', None),
        ('DspHighCutFrequency', '', None),
        ('DspHighCutNumTaps', '', None),
        ('DspHighCutFilterType', '', None),
        ('DspDelayCompensation', '', None),
        ('DspFilterDelay_µs', '', None),
        ('DisabledSubChannels', '', None),
        ('WaveformLength', '', int),
        ('AlignmentPt', '', None),
        ('ThreshVal', '', None),
        ('MinRetriggerSamples', '', None),
        ('SpikeRetriggerTime', '', None),
        ('DualThresholding', '', None),
        (r'Feature \w+ \d+', '', None),
        ('SessionUUID', '', None),
        ('FileUUID', '', None),
        ('CheetahRev', '', None),  # only for older versions of Cheetah
        ('ProbeName', '', None),
        ('OriginalFileName', '', None),
        ('TimeCreated', '', None),
        ('TimeClosed', '', None),
        ('ApplicationName', '', None),  # also include version number when present
        ('AcquisitionSystem', '', None),
        ('ReferenceChannel', '', None),
        ('NLX_Base_Class_Type', '', None)  # in version 4 and earlier versions of Cheetah
    ]

    # Filename and datetime may appear in header lines starting with # at
    # beginning of header or in later versions as a property. The exact format
    # used depends on the application name and its version as well as the
    # -FileVersion property.
    #
    # There are 4 styles understood by this code and the patterns used for parsing
    # the items within each are stored in a dictionary. Each dictionary is then
    # stored in main dictionary keyed by an abbreviation for the style.
    header_pattern_dicts = {
        # BML
        'bml': dict(
            datetime1_regex=r'## Time Opened: \(m/d/y\): (?P<date>\S+)'
                            r'  At Time: (?P<time>\S+)',
            filename_regex=r'## File Name: (?P<filename>\S+)',
            datetimeformat='%m/%d/%y %H:%M:%S.%f'),
        # Cheetah after version 1 and before version 5
        'bv5': dict(
            datetime1_regex=r'## Time Opened: \(m/d/y\): (?P<date>\S+)'
                            r'  At Time: (?P<time>\S+)',
            filename_regex=r'## File Name: (?P<filename>\S+)',
            datetimeformat='%m/%d/%Y %H:%M:%S.%f'),
        # Cheetah version 5 before and including v 5.6.4 as well as version 1
        'bv5.6.4': dict(
            datetime1_regex=r'## Time Opened \(m/d/y\): (?P<date>\S+)'
                            r'  \(h:m:s\.ms\) (?P<time>\S+)',
            datetime2_regex=r'## Time Closed \(m/d/y\): (?P<date>\S+)'
                            r'  \(h:m:s\.ms\) (?P<time>\S+)',
            filename_regex=r'## File Name (?P<filename>\S+)',
            datetimeformat='%m/%d/%Y %H:%M:%S.%f'),
        # Cheetah after v 5.6.4 and default for others such as Pegasus
        'def': dict(
            datetime1_regex=r'-TimeCreated (?P<date>\S+) (?P<time>\S+)',
            datetime2_regex=r'-TimeClosed (?P<date>\S+) (?P<time>\S+)',
            filename_regex=r'-OriginalFileName "?(?P<filename>\S+)"?',
            datetimeformat='%Y/%m/%d %H:%M:%S')
    }

    @staticmethod
    def build_for_file(filename):
        """
        Factory function to build NlxHeader for a given file.
        """

        with open(filename, 'rb') as f:
            txt_header = f.read(NlxHeader.HEADER_SIZE)
        txt_header = txt_header.strip(b'\x00').decode('latin-1')

        # must start with 8 # characters
        assert txt_header.startswith("########"),\
            'Neuralynx files must start with 8 # characters.'

        # find keys
        info = NlxHeader()
        for k1, k2, type_ in NlxHeader.txt_header_keys:
            pattern = r'-(?P<name>' + k1 + r')\s+(?P<value>[\S ]*)'
            matches = re.findall(pattern, txt_header)
            for match in matches:
                if k2 == '':
                    name = match[0]
                else:
                    name = k2
                value = match[1].rstrip(' ')
                if type_ is not None:
                    value = type_(value)
                info[name] = value

        # if channel_ids or s not in info then the filename is used
        name = os.path.splitext(os.path.basename(filename))[0]

        # convert channel ids
        if 'channel_ids' in info:
            chid_entries = re.findall(r'\w+', info['channel_ids'])
            info['channel_ids'] = [int(c) for c in chid_entries]
        else:
            info['channel_ids'] = [name]

        # convert channel names
        if 'channel_names' in info:
            name_entries = re.findall(r'\w+', info['channel_names'])
            if len(name_entries) == 1:
                info['channel_names'] = name_entries * len(info['channel_ids'])
            assert len(info['channel_names']) == len(info['channel_ids']), \
                'Number of channel ids does not match channel names.'
        else:
            info['channel_names'] = [name] * len(info['channel_ids'])

        # version and application name
        # older Cheetah versions with CheetahRev property
        if 'CheetahRev' in info:
            assert 'ApplicationName' not in info
            info['ApplicationName'] = 'Cheetah'
            app_version = info['CheetahRev']
        # new file version 3.4 does not contain CheetahRev property, but ApplicationName instead
        elif 'ApplicationName' in info:
            pattern = r'(\S*) "([\S ]*)"'
            match = re.findall(pattern, info['ApplicationName'])
            assert len(match) == 1, 'impossible to find application name and version'
            info['ApplicationName'], app_version = match[0]
        # BML Ncs file writing contained neither property, assume BML version 2
        else:
            info['ApplicationName'] = 'BML'
            app_version = "2.0"

        info['ApplicationVersion'] = distutils.version.LooseVersion(app_version)

        # convert bit_to_microvolt
        if 'bit_to_microVolt' in info:
            btm_entries = re.findall(r'\S+', info['bit_to_microVolt'])
            if len(btm_entries) == 1:
                btm_entries = btm_entries * len(info['channel_ids'])
            info['bit_to_microVolt'] = [float(e) * 1e6 for e in btm_entries]
            assert len(info['bit_to_microVolt']) == len(info['channel_ids']), \
                'Number of channel ids does not match bit_to_microVolt conversion factors.'

        if 'InputRange' in info:
            ir_entries = re.findall(r'\w+', info['InputRange'])
            if len(ir_entries) == 1:
                info['InputRange'] = [int(ir_entries[0])] * len(chid_entries)
            else:
                info['InputRange'] = [int(e) for e in ir_entries]
            assert len(info['InputRange']) == len(chid_entries), \
                'Number of channel ids does not match input range values.'

        # Format of datetime depends on app name, app version
        # :TODO: this works for current examples but is not likely actually related
        # to app version in this manner.
        an = info['ApplicationName']
        if an == 'Cheetah':
            av = info['ApplicationVersion']
            if av <= '2':  # version 1 uses same as older versions
                hpd = NlxHeader.header_pattern_dicts['bv5.6.4']
            elif av < '5':
                hpd = NlxHeader.header_pattern_dicts['bv5']
            elif av <= '5.6.4':
                hpd = NlxHeader.header_pattern_dicts['bv5.6.4']
            else:
                hpd = NlxHeader.header_pattern_dicts['def']
        elif an == 'BML':
            hpd = NlxHeader.header_pattern_dicts['bml']
        else:
            hpd = NlxHeader.header_pattern_dicts['def']

        # opening time
        dt1 = re.search(hpd['datetime1_regex'], txt_header).groupdict()
        info['recording_opened'] = datetime.datetime.strptime(
            dt1['date'] + ' ' + dt1['time'], hpd['datetimeformat'])

        # close time, if available
        if 'datetime2_regex' in hpd:
            dt2 = re.search(hpd['datetime2_regex'], txt_header).groupdict()
            info['recording_closed'] = datetime.datetime.strptime(
                dt2['date'] + ' ' + dt2['time'], hpd['datetimeformat'])

        return info

    def type_of_recording(self):
        """
        Determines type of recording in Ncs file with this header.

        RETURN:
            one of 'PRE4','BML','DIGITALLYNX','DIGITALLYNXSX','UNKNOWN'
        """

        if 'NLX_Base_Class_Type' in self:

            # older style standard neuralynx acquisition with rounded sampling frequency
            if self['NLX_Base_Class_Type'] == 'CscAcqEnt':
                return 'PRE4'

            # BML style with fractional frequency and microsPerSamp
            elif self['NLX_Base_Class_Type'] == 'BmlAcq':
                return 'BML'

            else:
                return 'UNKNOWN'

        elif 'HardwareSubSystemType' in self:

            # DigitalLynx
            if self['HardwareSubSystemType'] == 'DigitalLynx':
                return 'DIGITALLYNX'

            # DigitalLynxSX
            elif self['HardwareSubSystemType'] == 'DigitalLynxSX':
                return 'DIGITALLYNXSX'

        elif 'FileType' in self:

            if self['FileVersion'] in ['3.3', '3.4']:
                return self['AcquisitionSystem'].split()[1].upper()

            else:
                return 'UNKNOWN'

        else:
            return 'UNKNOWN'
