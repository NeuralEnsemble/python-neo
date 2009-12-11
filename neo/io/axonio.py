# -*- coding: utf-8 -*-
"""
axonio
==================

Classe for reading/writing data from pCLAMP and AxoScope 
files (.abf version 1 and 2), develloped by Molecular device/Axon technologies.

abf = Axon binary file

atf is a text file based from axon that can be read by AsciiIO.
but this file is less efficient.

This code is a port abfload and abf2load
written in Matlab by
Copyright (c) 2009, Forrest Collman 
                    fcollman@princeton.edu
Copyright (c) 2004, Harald Hentschke
and disponible here :
http://www.mathworks.com/matlabcentral/fileexchange/22114-abf2load

information on abf 1 and 2 format are disponible here:
http://www.moleculardevices.com/pages/software/developer_info.html


Classes
-------

AxonIO          - Classe for reading/writing data in abf axon files.

@author : sgarcia

"""

import struct
from baseio import BaseIO
from neo.core import *
from numpy import *
import re
import datetime


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
    
    **Usage**

    **Example**
    
    """
    
    is_readable        = True
    is_writable        = False
    is_object_readable = False
    is_object_writable = False
    has_header         = False
    is_streameable     = False
    read_params        = {}
    write_params       = {}
    level              = None
    nfiles             = 0
    name               = None
    objects            = []
    supported_types    = []
    
    def __init__(self ) :
        """
        
        **Arguments**
        
        """
        
        BaseIO.__init__(self)


    def read(self , *args, **kargs):
        """
        Read the file.
        Return a neo.Block by default
        See read_block for detail.
        
        You can also call read_segment if you assume that your file contain only
        one Segment.
        """
        return self.read_block( *args , **kargs)
    
    def read_block(self, filename = '', ):
        """
        **Arguments**
            filename : filename
            TODO
        """
        BLOCKSIZE=512
        
        print self.read_header(filename = filename)
        
        block = Block()
        return block



    def read_header(self, filename = None):
        """
        read the header of the file
        """
        fid = struct_file(filename,'rb')
        fFileSignature =  fid.read(4)
        
        if fFileSignature == 'ABF ' :
            headerDescription = headerDescriptionV1
        elif fFileSignature == 'ABF2' :
            headerDescription = headerDescriptionV2
        else :
            return None
            
        # construct dict
        header = { }
        for name, offset , format in headerDescription :
            val = fid.read_f(format , offset = offset)
            if len(val) == 1:
                header[name] = val[0]
            else :
                header[name] = array(val)
        
        # correction of version number and starttime
        if fFileSignature == 'ABF ' :
            header['lFileStartTime'] =  header['lFileStartTime'] +  header['nFileStartMillisecs']*.001
        elif fFileSignature == 'ABF2' :
            n = header['fFileVersionNumber']
            header['fFileVersionNumber'] = n[3]+0.1*n[2]+0.01*n[1]+0.001*n[0]
            header['lFileStartTime'] = header['uFileStartTimeMS']*.001
        
        
        if header['fFileVersionNumber'] >= 2. :
            # in abf2 some info are in other place
            
            # sections 
            sections = { }
            for s,sectionName in enumerate(sectionNames) :
                uBlockIndex,uBytes,llNumEntries= fid.read_f( 'IIl' , offset = 76 + s * 16 )
                sections[sectionName] = { }
                sections[sectionName]['uBlockIndex'] = uBlockIndex
                sections[sectionName]['uBytes'] = uBytes
                sections[sectionName]['llNumEntries'] = llNumEntries
            
            fid.seek(sections['StringsSection']['uBlockIndex']*BLOCKSIZE)
            bigString = fid.read(sections['StringsSection']['uBytes'])
            
            #hack
            goodstart = bigString.lower().find('clampex')
            if goodstart == -1 :
                goodstart = bigString.lower().find('axoscope')
            
            bigString = bigString[goodstart:]
            print bigString
            strings = bigString.split('\x00')
            print strings
            
            header['nADCSamplingSeq'] = [ ]
            header['recChNames'] = [ ]
            header['recChUnits'] = [ ]
            
            print sections['ADCSection']['llNumEntries']
            for i in range(sections['ADCSection']['llNumEntries']) :
                #  read ADCInfo
                fid.seek(sections['ADCSection']['uBlockIndex']*\
                            BLOCKSIZE+sections['ADCSection']['uBytes']*i)
                ADCInfo = { }
                for name, format in ADCInfoDescription :
                    val = fid.read_f(format )
                    if len(val) == 1:
                        ADCInfo[name] = val[0]
                    else :
                        ADCInfo[name] = array(val)                
                
                j = ADCInfo['nADCNum']
                if j != i:
                    print 'OULALA', i, j
                print ADCInfo['lADCChannelNameIndex'] , strings[ADCInfo['lADCChannelNameIndex']-1]
                print ADCInfo['lADCUnitsIndex'] , strings[ADCInfo['lADCUnitsIndex']-1]
                header['nADCSamplingSeq'].append( ADCInfo['nADCNum'] ) 
                header['recChNames'].append( strings[ADCInfo['lADCChannelNameIndex']-1] )
                header['recChUnits'].append( strings[ADCInfo['lADCUnitsIndex']-1] )

            
            
            
            
            
            
            
            
            
            

            
        
        
        print header
        

        
        
        
        fid.close()


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


