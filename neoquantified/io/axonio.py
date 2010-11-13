# -*- coding: utf-8 -*-
"""

Classe for reading data from pCLAMP and AxoScope 
files (.abf version 1 and 2), develloped by Molecular device/Axon technologies.

- abf = Axon binary file
- atf is a text file based from axon that could be read by AsciiIO (but this file is less efficient.)


This code is a port abfload and abf2load
written in Matlab (BSD licence) by :
 - Copyright (c) 2009, Forrest Collman ,fcollman@princeton.edu
 - Copyright (c) 2004, Harald Hentschke
and disponible here : http://www.mathworks.com/matlabcentral/fileexchange/22114-abf2load

information on abf 1 and 2 format are disponible here : http://www.moleculardevices.com/pages/software/developer_info.html

This file support old (ABF1) and new (ABF2) format.

Supported : Read


@author : sgarcia

"""

import struct
from baseio import BaseIO
#from neo.core import *
from ..core import *
from numpy import *
from numpy import memmap
import re
import datetime


# TODO
# - gerer le t_start pour sweep
# verifier avec BP le event (Tags)

class struct_file(file):
    def read_f(self, format , offset = None):
        if offset is not None:
            self.seek(offset)
        return struct.unpack(format , self.read(struct.calcsize(format)))

    def write_f(self, format , offset = None , *args ):
        if offset is not None:
            self.seek(offset)
        self.write( struct.pack( format , *args ) )

class AxonIO(BaseIO):
    """
    Classe for reading/writing data from axon binary file(.abf)
    Read ABF1 (clampfit <=9) and ABF2 (clampfit >10)
    
    **Example**

    #read a file
    io = AxonIO(filename = 'myfile.abf')
    blck = io.read() # read the entire file
    
    blck contains one or several Segments
    Segments contains AnaloSignals and/or Events
    """
    
    is_readable        = True
    is_writable        = False
    
    supported_objects  = [Block , Segment , AnalogSignal , Event]
    readable_objects   = [Block]
    writeable_objects  = []  

    has_header         = False
    is_streameable     = False
    
    read_params        = { Block : [] }
    write_params       = None

    name               = None
    extensions         = [ 'abf' ]
    
    mode = 'file'
    
    
    def __init__(self , filename = None) :
        """
        This class read a abf file.
        
        **Arguments**
        
            filename : the filename to read
        
        """
        BaseIO.__init__(self)
        self.filename = filename

    def read(self , *args, **kargs):
        """
        Read the file.
        Return a neo.Block by default
        See read_block for detail.
        
        """
        return self.read_block( *args , **kargs)
    
    def read_block(self, ):
        """
        Read a abf file.
        All possible mode are possible :
            - event-driven variable-length mode (mode 1) -> return several Segment in the Block
            - event-driven fixed-length mode (mode 2 or 5) -> return several Segment in the Block
            - gap free mode -> return one Segment in the Block
        
        **Arguments**
            no argument
        """
        
        def reformat_integer_V1(data, nbchannel , header):
            """
            reformat when dtype is int16 for ABF version 1
            """
            for i in range(nbchannel):
                data[:,i] /= header['fInstrumentScaleFactor'][i]
                data[:,i] /= header['fSignalGain'][i]
                data[:,i] /= header['fADCProgrammableGain'][i]
                if header['nTelegraphEnable'][i] :
                    data[:,i] /= header['fTelegraphAdditGain'][i]
                data[:,i] *= header['fADCRange']
                data[:,i] /= header['lADCResolution']
                data[:,i] += header['fInstrumentOffset'][i]
                data[:,i] -= header['fSignalOffset'][i]
            
        def reformat_integer_V2(data, nbchannel , header):
            """
            reformat when dtype is int16 for ABF version 2
            """
            for i in range(nbchannel):
                data[:,i] /= header['listADCInfo'][i]['fInstrumentScaleFactor']
                data[:,i] /= header['listADCInfo'][i]['fSignalGain']
                data[:,i] /= header['listADCInfo'][i]['fADCProgrammableGain']
                if header['listADCInfo'][i]['nTelegraphEnable'] :
                    data[:,i] /= header['listADCInfo'][i]['fTelegraphAdditGain']
                data[:,i] *= header['protocol']['fADCRange']
                data[:,i] /= header['protocol']['lADCResolution']
                data[:,i] += header['listADCInfo'][i]['fInstrumentOffset']
                data[:,i] -= header['listADCInfo'][i]['fSignalOffset']
        
        def clean_string(s):
            while s.endswith('\x00') :
                s = s[:-1]
            while s.endswith(' ') :
                s = s[:-1]
            return s
            
        
        
        header = self.read_header()
        version = header['fFileVersionNumber']
        
        #print 'version' , version
        
        # date and time
        if version <2. :
            YY = 1900
            MM = 01
            DD = 01
            hh = int(header['lFileStartTime']/3600.)
            mm = int((header['lFileStartTime']-hh*3600)/60)
            ss = header['lFileStartTime']-hh*3600-mm*60
            ms = int(mod(ss,1)*1e6)
            ss = int(ss)            
        elif version >=2. :
            YY = int(header['uFileStartDate']/10000)
            MM = int((header['uFileStartDate']-YY*10000)/100)
            DD = int(header['uFileStartDate']-YY*10000-MM*100)
            hh = int(header['uFileStartTimeMS']/1000./3600.)
            mm = int((header['uFileStartTimeMS']/1000.-hh*3600)/60)
            ss = header['uFileStartTimeMS']/1000.-hh*3600-mm*60
            ms = int(mod(ss,1)*1e6)
            ss = int(ss)        
        filedatetime = datetime.datetime(  YY , MM , DD , hh , mm , ss , ms)
        
        # file format
        if header['nDataFormat'] == 0 :
            dt = dtype('i2')
        elif header['nDataFormat'] == 1 :
            dt = dtype('f4')
        
        if version <2. :
            nbchannel = header['nADCNumChannels']
            headOffset = header['lDataSectionPtr']*BLOCKSIZE+header['nNumPointsIgnored']*dt.itemsize
            totalsize = header['lActualAcqLength']
        elif version >=2. :
            nbchannel = header['sections']['ADCSection']['llNumEntries']
            headOffset = header['sections']['DataSection']['uBlockIndex']*BLOCKSIZE
            totalsize = header['sections']['DataSection']['llNumEntries']
        
        data = memmap(self.filename , dt  , 'r', 
                          shape = (totalsize,) , offset = headOffset)
        
        # 3 possible modes
        if version <2. :
            mode = header['nOperationMode']
        elif version >=2. :
            mode = header['protocol']['nOperationMode']
        #print 'mode' , mode
        if (mode == 1) or (mode == 2) or  (mode == 5):
            # event-driven variable-length mode (mode 1)
            # event-driven fixed-length mode (mode 2 or 5)
            
            # read sweep pos
            nbsweep = header['lActualEpisodes']
            if version <2. :
                nbsweep2 = header['lSynchArraySize']
                offsetSweep = header['lSynchArrayPtr']*BLOCKSIZE
            elif version >=2. :
                nbsweep2 = header['sections']['SynchArraySection']['llNumEntries']
                offsetSweep = header['sections']['SynchArraySection']['uBlockIndex']*BLOCKSIZE
            sweepArray = memmap(self.filename , 'i4'  , 'r',
                                        shape = (nbsweep2, 2),
                                        offset = offsetSweep )            
            
            # read subset of data
            list_data = [ ]
            pos = 0
            for j in range(nbsweep) :
                length = sweepArray[j,1]
                
                if version <2. :
                    fSynchTimeUnit = header['fSynchTimeUnit']
                elif version >=2. :
                    fSynchTimeUnit = header['protocol']['fSynchTimeUnit']
                
                if (fSynchTimeUnit != 0) and (mode == 1) :
                    length /= fSynchTimeUnit
                subdata  = data[pos:pos+length]
                pos += length
                subdata = subdata.reshape( (subdata.size/nbchannel, nbchannel )).astype('f')
                if dt == dtype('i2'):
                    if version <2. :
                        reformat_integer_V1(subdata, nbchannel , header)
                    elif version >=2. :
                        reformat_integer_V2(subdata, nbchannel , header)
                list_data.append(subdata)
            
            # construct block
            # one sweep = one segment in a block
            block = Block()
            block.datetime = filedatetime
            for j in range(nbsweep) :
                seg = Segment()
                seg.num = j
                for i in range(nbchannel):
                    if version <2. :
                        sampling_rate = 1./(header['fADCSampleInterval']*nbchannel*1.e-6)
                        name = header['sADCChannelName'][i]
                        unit = header['sADCUnits'][i]
                        num = header['nADCPtoLChannelMap'][i]
                    elif version >=2. :
                        sampling_rate = 1.e6/header['protocol']['fADCSequenceInterval']
                        name = header['listADCInfo'][i]['recChNames']
                        unit = header['listADCInfo'][i]['recChUnits']
                        num = header['listADCInfo'][i]['nADCNum']
                    anaSig = AnalogSignal( signal = list_data[j][:,i],
                                            sampling_rate = sampling_rate ,
                                            t_start = 0)
                    anaSig.name = name
                    anaSig.unit = unit
                    anaSig.channel = num
                    seg._analogsignals.append( anaSig )
                block._segments.append(seg)

        elif (mode == 3) :
            # gap free mode
            m = data.size%nbchannel
            if m != 0 : data = data[:-m]
            data = data.reshape( (data.size/nbchannel, nbchannel)).astype('f')
            if dt == dtype('i2'):
                if version <2. :
                    reformat_integer_V1(data, nbchannel , header)
                elif version >=2. :
                    reformat_integer_V2(data, nbchannel , header)                
            
            # one segment in one block
            block = Block()
            seg = Segment()
            seg.datetime = filedatetime
            for i in range(nbchannel):
                if version <2. :
                    sampling_rate = 1./(header['fADCSampleInterval']*nbchannel*1.e-6)
                    name = header['sADCChannelName'][i]
                    unit = header['sADCUnits'][i]
                    num = header['nADCPtoLChannelMap'][i]
                elif version >=2. :
                    sampling_rate = 1.e6/header['protocol']['fADCSequenceInterval']
                    name = header['listADCInfo'][i]['recChNames']
                    unit = header['listADCInfo'][i]['recChUnits']
                    num = header['listADCInfo'][i]['nADCNum']
                anaSig = AnalogSignal( signal = data[:,i],
                                        sampling_rate = sampling_rate ,
                                        t_start = 0)
                anaSig.name = name
                anaSig.unit = unit
                anaSig.channel = num
            seg._analogsignals.append( anaSig )
            for i,tag in enumerate(header['listTag']) :
                event = Event(  )
                event.time = tag['lTagTime']/sampling_rate
                event.name = clean_string(tag['sComment'])
                event.num = i
                event.type = tag['nTagType']
                seg._events.append( event )
            block._segments.append(seg)

        return block


    def read_header(self, ):
        """
        read the header of the file
        
        The startegy differ here from the orignal script under Matlab.
        In the original script for ABF2, it complete the header with informations
        that are located in other strutures.
        
        In ABF2 this function return header with sub dict :
            listADCInfo
            protocole
            tags
        that contain more information.
        """
        fid = struct_file(self.filename,'rb')
        
        # version
        fFileSignature =  fid.read(4)
        if fFileSignature == 'ABF ' :
            headerDescription = headerDescriptionV1
        elif fFileSignature == 'ABF2' :
            headerDescription = headerDescriptionV2
        else :
            return None
            
        # construct dict
        header = { }
        for key, offset , format in headerDescription :
            val = fid.read_f(format , offset = offset)
            if len(val) == 1:
                header[key] = val[0]
            else :
                header[key] = array(val)
        
        # correction of version number and starttime
        if fFileSignature == 'ABF ' :
            header['lFileStartTime'] =  header['lFileStartTime'] +  header['nFileStartMillisecs']*.001
        elif fFileSignature == 'ABF2' :
            n = header['fFileVersionNumber']
            header['fFileVersionNumber'] = n[3]+0.1*n[2]+0.01*n[1]+0.001*n[0]
            header['lFileStartTime'] = header['uFileStartTimeMS']*.001
        
        if header['fFileVersionNumber'] < 2. :
            # tags
            listTag = [ ]
            for i in range(header['lNumTagEntries']) :
                fid.seek(header['lTagSectionPtr']+i*64)
                tag = { }
                for key, format in TagInfoDescription :
                    val = fid.read_f(format )
                    if len(val) == 1:
                        tag[key] = val[0]
                    else :
                        tag[key] = array(val)
                listTag.append(tag)
            header['listTag'] = listTag
        
        elif header['fFileVersionNumber'] >= 2. :
            # in abf2 some info are in other place
            
            # sections 
            sections = { }
            for s,sectionName in enumerate(sectionNames) :
                uBlockIndex,uBytes,llNumEntries= fid.read_f( 'IIl' , offset = 76 + s * 16 )
                sections[sectionName] = { }
                sections[sectionName]['uBlockIndex'] = uBlockIndex
                sections[sectionName]['uBytes'] = uBytes
                sections[sectionName]['llNumEntries'] = llNumEntries
            header['sections'] = sections
            
            
            # strings sections
            # hack for reading channels names and units
            fid.seek(sections['StringsSection']['uBlockIndex']*BLOCKSIZE)
            bigString = fid.read(sections['StringsSection']['uBytes'])
            goodstart = bigString.lower().find('clampex')
            if goodstart == -1 :
                goodstart = bigString.lower().find('axoscope')
            
            bigString = bigString[goodstart:]
            strings = bigString.split('\x00')
            
            
            # ADC sections
            header['listADCInfo'] = [ ]
            for i in range(sections['ADCSection']['llNumEntries']) :
                #  read ADCInfo
                fid.seek(sections['ADCSection']['uBlockIndex']*\
                            BLOCKSIZE+sections['ADCSection']['uBytes']*i)
                ADCInfo = { }
                for key, format in ADCInfoDescription :
                    val = fid.read_f(format )
                    if len(val) == 1:
                        ADCInfo[key] = val[0]
                    else :
                        ADCInfo[key] = array(val)
                ADCInfo['recChNames'] = strings[ADCInfo['lADCChannelNameIndex']-1]
                ADCInfo['recChUnits'] = strings[ADCInfo['lADCUnitsIndex']-1]
                
                header['listADCInfo'].append( ADCInfo )
        
            # protocol sections
            protocol = { }
            fid.seek(sections['ProtocolSection']['uBlockIndex']*BLOCKSIZE)
            for key, format in protocolInfoDescription :
                val = fid.read_f(format )
                if len(val) == 1:
                    protocol[key] = val[0]
                else :
                    protocol[key] = array(val)
            header['protocol'] = protocol
            
            # tags
            listTag = [ ]
            for i in range(sections['TagSection']['llNumEntries']) :
                fid.seek(sections['TagSection']['uBlockIndex']*\
                            BLOCKSIZE+sections['TagSection']['uBytes']*i)
                tag = { }
                for key, format in TagInfoDescription :
                    val = fid.read_f(format )
                    if len(val) == 1:
                        tag[key] = val[0]
                    else :
                        tag[key] = array(val)
                listTag.append(tag)
                
            header['listTag'] = listTag
                
            
        fid.close()
        
        return header


BLOCKSIZE = 512

headerDescriptionV1= [
         ('fFileSignature',0,'4s'),
         ('fFileVersionNumber',4,'f' ),
         ('nOperationMode',8,'h' ),
         ('lActualAcqLength',10,'i' ),
         ('nNumPointsIgnored',14,'h' ),
         ('lActualEpisodes',16,'i' ),
         ('lFileStartTime',24,'i' ),
         ('lDataSectionPtr',40,'i' ),
         ('lTagSectionPtr',44,'i' ),
         ('lNumTagEntries',48,'i' ),
         ('lSynchArrayPtr',92,'i' ),
         ('lSynchArraySize',96,'i' ),
         ('nDataFormat',100,'h' ),
         ('nADCNumChannels', 120, 'h'),
         ('fADCSampleInterval',122,'f'),
         ('fSynchTimeUnit',130,'f' ),
         ('lNumSamplesPerEpisode',138,'i' ),
         ('lPreTriggerSamples',142,'i' ),
         ('lEpisodesPerRun',146,'i' ),
         ('fADCRange', 244, 'f' ),
         ('lADCResolution', 252, 'i'),
         ('nFileStartMillisecs', 366, 'h'),
         ('nADCPtoLChannelMap', 378, '16h'),
         ('nADCSamplingSeq', 410, '16h'),
         ('sADCChannelName',442, '10s'*16),
         ('sADCUnits',602, '8s'*16) ,
         ('fADCProgrammableGain', 730, '16f'),
         ('fInstrumentScaleFactor', 922, '16f'),
         ('fInstrumentOffset', 986, '16f'),
         ('fSignalGain', 1050, '16f'),
         ('fSignalOffset', 1114, '16f'),
         ('nTelegraphEnable',4512, '16h'),
         ('fTelegraphAdditGain',4576,'16f'),
         ]


headerDescriptionV2 =[
         ('fFileSignature',0,'4s' ),
         ('fFileVersionNumber',4,'4b') , 
         ('uFileInfoSize',8,'I' ) ,
         ('lActualEpisodes',12,'I' ) ,
         ('uFileStartDate',16,'I' ) ,
         ('uFileStartTimeMS',20,'I' ) ,
         ('uStopwatchTime',24,'I' ) ,
         ('nFileType',28,'H' ) ,
         ('nDataFormat',30,'H' ) ,
         ('nSimultaneousScan',32,'H' ) ,
         ('nCRCEnable',34,'H' ) ,
         ('uFileCRC',36,'I' ) ,
         ('FileGUID',40,'I' ) ,
         ('uCreatorVersion',56,'I' ) ,
         ('uCreatorNameIndex',60,'I' ) ,
         ('uModifierVersion',64,'I' ) ,
         ('uModifierNameIndex',68,'I' ) ,
         ('uProtocolPathIndex',72,'I' ) ,
         ]


sectionNames= ['ProtocolSection',
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
         ('nOperationMode','h'),
         ('fADCSequenceInterval','f'),
         ('bEnableFileCompression','b'),
         ('sUnused1','3s'),
         ('uFileCompressionRatio','I'),
         ('fSynchTimeUnit','f'),
         ('fSecondsPerRun','f'),
         ('lNumSamplesPerEpisode','i'),
         ('lPreTriggerSamples','i'),
         ('lEpisodesPerRun','i'),
         ('lRunsPerTrial','i'),
         ('lNumberOfTrials','i'),
         ('nAveragingMode','h'),
         ('nUndoRunCount','h'),
         ('nFirstEpisodeInRun','h'),
         ('fTriggerThreshold','f'),
         ('nTriggerSource','h'),
         ('nTriggerAction','h'),
         ('nTriggerPolarity','h'),
         ('fScopeOutputInterval','f'),
         ('fEpisodeStartToStart','f'),
         ('fRunStartToStart','f'),
         ('lAverageCount','i'),
         ('fTrialStartToStart','f'),
         ('nAutoTriggerStrategy','h'),
         ('fFirstRunDelayS','f'),
         ('nChannelStatsStrategy','h'),
         ('lSamplesPerTrace','i'),
         ('lStartDisplayNum','i'),
         ('lFinishDisplayNum','i'),
         ('nShowPNRawData','h'),
         ('fStatisticsPeriod','f'),
         ('lStatisticsMeasurements','i'),
         ('nStatisticsSaveStrategy','h'),
         ('fADCRange','f'),
         ('fDACRange','f'),
         ('lADCResolution','i'),
         ('lDACResolution','i'),
         ('nExperimentType','h'),
         ('nManualInfoStrategy','h'),
         ('nCommentsEnable','h'),
         ('lFileCommentIndex','i'),
         ('nAutoAnalyseEnable','h'),
         ('nSignalType','h'),
         ('nDigitalEnable','h'),
         ('nActiveDACChannel','h'),
         ('nDigitalHolding','h'),
         ('nDigitalInterEpisode','h'),
         ('nDigitalDACChannel','h'),
         ('nDigitalTrainActiveLogic','h'),
         ('nStatsEnable','h'),
         ('nStatisticsClearStrategy','h'),
         ('nLevelHysteresis','h'),
         ('lTimeHysteresis','i'),
         ('nAllowExternalTags','h'),
         ('nAverageAlgorithm','h'),
         ('fAverageWeighting','f'),
         ('nUndoPromptStrategy','h'),
         ('nTrialTriggerSource','h'),
         ('nStatisticsDisplayStrategy','h'),
         ('nExternalTagType','h'),
         ('nScopeTriggerOut','h'),
         ('nLTPType','h'),
         ('nAlternateDACOutputState','h'),
         ('nAlternateDigitalOutputState','h'),
         ('fCellID','3f'),
         ('nDigitizerADCs','h'),
         ('nDigitizerDACs','h'),
         ('nDigitizerTotalDigitalOuts','h'),
         ('nDigitizerSynchDigitalOuts','h'),
         ('nDigitizerType','h'),
         ]


ADCInfoDescription = [
         ('nADCNum','h'),
         ('nTelegraphEnable','h'),
         ('nTelegraphInstrument','h'),
         ('fTelegraphAdditGain','f'),
         ('fTelegraphFilter','f'),
         ('fTelegraphMembraneCap','f'),
         ('nTelegraphMode','h'),
         ('fTelegraphAccessResistance','f'),
         ('nADCPtoLChannelMap','h'),
         ('nADCSamplingSeq','h'),
         ('fADCProgrammableGain','f'),
         ('fADCDisplayAmplification','f'),
         ('fADCDisplayOffset','f'),
         ('fInstrumentScaleFactor','f'),
         ('fInstrumentOffset','f'),
         ('fSignalGain','f'),
         ('fSignalOffset','f'),
         ('fSignalLowpassFilter','f'),
         ('fSignalHighpassFilter','f'),
         ('nLowpassFilterType','b'),
         ('nHighpassFilterType','b'),
         ('fPostProcessLowpassFilter','f'),
         ('nPostProcessLowpassFilterType','c'),
         ('bEnabledDuringPN','b'),
         ('nStatsChannelPolarity','h'),
         ('lADCChannelNameIndex','i'),
         ('lADCUnitsIndex','i'),
         ]

TagInfoDescription = [
       ('lTagTime','i'),
       ('sComment','56s'),
       ('nTagType','h'),
       ('nVoiceTagNumber_or_AnnotationIndex','h'),
       ]


