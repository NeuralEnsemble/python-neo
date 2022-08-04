import math
import numpy as np


class NcsSections:
    """
    Contains information regarding the contiguous sections of records in an Ncs file.
    Methods of NcsSectionsFactory perform parsing of this information from an Ncs file and
    produce these where the sections are discontiguous in time and in temporal order.

    TODO: This class will likely need __ne__ to be useful in
    more sophisticated segment construction algorithms.

    """
    def __init__(self):
        self.sects = []
        self.sampFreqUsed = 0  # actual sampling frequency of samples
        self.microsPerSampUsed = 0  # microseconds per sample

    def __eq__(self, other):
        samp_eq = self.sampFreqUsed == other.sampFreqUsed
        micros_eq = self.microsPerSampUsed == other.microsPerSampUsed
        sects_eq = self.sects == other.sects
        return (samp_eq and micros_eq and sects_eq)

    def __hash__(self):
        return (f'{self.sampFreqUsed};{self.microsPerSampUsed};'
               f'{[s.__hash__() for s in self.sects]}').__hash__()

    def is_equivalent(self, other, rel_tol=0, abs_tol=0):
        if len(self.sects) != len(other.sects):
            return False
        else:
            # do not check for gaps if only a single section is present
            for sec_id in range(len(self.sects) - 1):
                if not self.sects[sec_id].is_equivalent(
                        other.sects[sec_id], rel_tol=rel_tol, abs_tol=abs_tol):
                    return False
            return True


class NcsSection:
    """
    Information regarding a single contiguous section or group of records in an Ncs file.

    Model is that times are closed on the left and open on the right. Record
    numbers are closed on both left and right, that is, inclusive of the last record.

    endTime should never be set less than startTime for comparison functions to work
    properly, though this is not enforced.
    """

    _RECORD_SIZE = 512  # nb sample per signal record

    def __init__(self):
        self.startRec = -1  # index of starting record
        self.startTime = -1  # starttime of first record
        self.endRec = -1  # index of last record (inclusive)
        self.endTime = -1   # end time of last record, that is, the end time of the last
                            # sampling period contained in the last record of the section

    def __init__(self, sb, st, eb, et, ns):
        self.startRec = sb
        self.startTime = st
        self.endRec = eb
        self.endTime = et
        self.n_samples = ns

    def __eq__(self, other):
        return (self.startRec == other.startRec
                and self.startTime == other.startTime
                and self.endRec == other.endRec
                and self.endTime == other.endTime
                and self.n_samples == other.n_samples)

    def __hash__(self):
        s = f'{self.startRec};{self.startTime};{self.endRec};{self.endTime};{self.n_samples}'
        return s.__hash__()

    def is_equivalent(self, other, rel_tol=0, abs_tol=0):
        eq_start = math.isclose(self.startTime, other.startTime, rel_tol=rel_tol, abs_tol=abs_tol)
        eq_end = math.isclose(self.endTime, other.endTime, rel_tol=rel_tol, abs_tol=abs_tol)
        return eq_start & eq_end


    def before_time(self, rhb):
        """
        Determine if this section is completely before another section in time.
        """
        return self.endTime < rhb.startTime

    def overlaps_time(self, rhb):
        """
        Determine if this section overlaps another in time.
        """
        return self.startTime <= rhb.endTime and self.endTime >= rhb.startTime

    def after_time(self, rhb):
        """
        Determine if this section is completely after another section in time.
        """
        return self.startTime >= rhb.endTime


class NcsSectionsFactory:
    """
    Class for factory methods which perform parsing of contiguous sections of records
    in Ncs files.

    Model for times is that times are rounded to nearest microsecond. Times
    from start of a sample until just before the next sample are included,
    that is, closed lower bound and open upper bound on intervals. A
    channel with no samples is empty and contains no time intervals.

    Moved here since algorithm covering all 3 header styles and types used is
    more complicated.
    """

    _maxGapSampFrac = 0.2  # maximum fraction of a sampling interval between predicted
                           # and actual record timestamps still considered within one section

    @staticmethod
    def get_freq_for_micros_per_samp(micros):
        """
        Compute fractional sampling frequency, given microseconds per sample.
        """
        return 1e6 / micros

    @staticmethod
    def get_micros_per_samp_for_freq(sampFr):
        """
        Calculate fractional microseconds per sample, given the sampling frequency (Hz).
        """
        return 1e6 / sampFr

    @staticmethod
    def calc_sample_time(sampFr, startTime, posn):
        """
        Calculate time rounded to microseconds for sample given frequency,
        start time, and sample position.
        """
        return round(startTime + NcsSectionsFactory.get_micros_per_samp_for_freq(sampFr) * posn)

    @staticmethod
    def _parseGivenActualFrequency(ncsMemMap, ncsSects, chanNum, reqFreq, blkOnePredTime):
        """
        Parse sections in memory mapped file when microsPerSampUsed and sampFreqUsed are known,
        filling in an NcsSections object.

        Parameters
        ----------
        ncsMemMap:
            memmap of Ncs file
        ncsSections:
            NcsSections with actual sampFreqUsed correct, first NcsSection with proper startSect
            and startTime already added.
        chanNum:
            channel number that should be present in all records
        reqFreq:
            rounded frequency that all records should contain
        blkOnePredTime:
            predicted starting time of second record in block

        Returns
        -------
        NcsSections object with block locations marked
        """
        startBlockPredTime = blkOnePredTime
        blk_len = 0
        curBlock = ncsSects.sects[0]
        for recn in range(1, ncsMemMap.shape[0]):
            timestamp = ncsMemMap['timestamp'][recn]
            channel_id = ncsMemMap['channel_id'][recn]
            sample_rate = ncsMemMap['sample_rate'][recn]
            nb_valid = ncsMemMap['nb_valid'][recn]

            if channel_id != chanNum or sample_rate != reqFreq:
                raise IOError('Channel number or sampling frequency changed in ' +
                              'records within file')
            predTime = NcsSectionsFactory.calc_sample_time(ncsSects.sampFreqUsed,
                                                           startBlockPredTime, blk_len)
            nValidSamps = nb_valid
            if timestamp != predTime:
                curBlock.endRec = recn - 1
                curBlock.endTime = predTime
                curBlock.n_samples = blk_len
                curBlock = NcsSection(recn, timestamp, -1, -1, -1)
                ncsSects.sects.append(curBlock)
                startBlockPredTime = NcsSectionsFactory.calc_sample_time(
                    ncsSects.sampFreqUsed,
                    timestamp,
                    nValidSamps)
                blk_len = 0
            else:
                blk_len += nValidSamps

        curBlock.endRec = ncsMemMap.shape[0] - 1
        endTime = NcsSectionsFactory.calc_sample_time(ncsSects.sampFreqUsed,
                                                      startBlockPredTime,
                                                      blk_len)
        curBlock.endTime = endTime

        return ncsSects

    @staticmethod
    def _buildGivenActualFrequency(ncsMemMap, actualSampFreq, reqFreq):
        """
        Build NcsSections object for file given actual sampling frequency.

        Requires that frequency in each record agrees with requested frequency. This is
        normally obtained by rounding the header frequency; however, this value may be different
        from the rounded actual frequency used in the recording, since the underlying
        requirement in older Ncs files was that the number of microseconds per sample in the
        records is the inverse of the sampling frequency stated in the header truncated to
        whole microseconds.

        Parameters
        ----------
        ncsMemMap:
            memmap of Ncs file
        actualSampFreq:
            actual sampling frequency used
        reqFreq:
            frequency to require in records

        Returns
        -------
            NcsSections object
        """
        # check frequency in first record
        if ncsMemMap['sample_rate'][0] != reqFreq:
            raise IOError("Sampling frequency in first record doesn't agree with header.")
        chanNum = ncsMemMap['channel_id'][0]

        secs = NcsSections()
        secs.sampFreqUsed = actualSampFreq
        secs.microsPerSampUsed = NcsSectionsFactory.get_micros_per_samp_for_freq(actualSampFreq)

        # check if file is one block of records, which is often the case, and avoid full parse
        lastBlkI = ncsMemMap.shape[0] - 1
        ts0 = ncsMemMap['timestamp'][0]
        nb0 = ncsMemMap['nb_valid'][0]
        predLastBlockStartTime = NcsSectionsFactory.calc_sample_time(actualSampFreq, ts0,
                                                                     NcsSection._RECORD_SIZE *
                                                                     lastBlkI)
        lts = ncsMemMap['timestamp'][lastBlkI]
        lnb = ncsMemMap['nb_valid'][lastBlkI]
        if ncsMemMap['channel_id'][lastBlkI] == chanNum and \
                ncsMemMap['sample_rate'][lastBlkI] == reqFreq and \
                lts == predLastBlockStartTime:
            lastBlkEndTime = NcsSectionsFactory.calc_sample_time(actualSampFreq, lts, lnb)
            n_samples = NcsSection._RECORD_SIZE * lastBlkI
            curBlock = NcsSection(0, ts0, lastBlkI, lastBlkEndTime, n_samples)

            secs.sects.append(curBlock)
            return secs

        # otherwise need to scan looking for breaks
        else:
            blkOnePredTime = NcsSectionsFactory.calc_sample_time(actualSampFreq, ts0, nb0)
            curBlock = NcsSection(0, ts0, -1, -1, -1)
            secs.sects.append(curBlock)
            return NcsSectionsFactory._parseGivenActualFrequency(ncsMemMap, secs, chanNum, reqFreq,
                                                                 blkOnePredTime)

    @staticmethod
    def _parseForMaxGap(ncsMemMap, ncsSects, maxGapLen):
        """
        Parse blocks of records from file, allowing a maximum gap in timestamps between records
        in sections. Estimates frequency being used based on timestamps.

        Parameters
        ----------
        ncsMemMap:
            memmap of Ncs file
        ncsSects:
            NcsSections object with sampFreqUsed set to nominal frequency to use in computing time
            for samples (Hz)
        maxGapLen:
            maximum difference within a block between predicted time of start of record and
            recorded time

        Returns
        -------
            NcsSections object with sampFreqUsed and microsPerSamp set based on estimate from
            largest block
        """

        chanNum = ncsMemMap['channel_id'][0]
        recFreq = ncsMemMap['sample_rate'][0]

        # check for consistent channel_ids and sampling rates
        ncsMemMap['channel_id']
        if not (ncsMemMap['channel_id'] == chanNum).all():
            raise IOError('Channel number changed in records within file')

        if not all(ncsMemMap['sample_rate'] == recFreq):
            raise IOError('Sampling frequency changed in records within file')

        # find most frequent number of samples
        exp_nb_valid = np.argmax(np.bincount(ncsMemMap['nb_valid']))
        # detect records with incomplete number of samples
        gap_rec_ids = list(np.where(ncsMemMap['nb_valid'] != exp_nb_valid)[0])

        rec_duration = 1e6 / ncsSects.sampFreqUsed * ncsMemMap['nb_valid']
        pred_times = np.rint(ncsMemMap['timestamp'] + rec_duration).astype(np.int64)
        max_pred_times = pred_times + maxGapLen
        # data records that start later than the predicted time (including the
        # maximal accepted gap length) are considered delayed and a gap is
        # registered.
        delayed_recs = list(np.where(max_pred_times[:-1] < ncsMemMap['timestamp'][1:])[0])
        gap_rec_ids.extend(delayed_recs)

        # cleaning extracted gap ids
        # last record can not be the beginning of a gap
        last_rec_id = len(ncsMemMap['timestamp']) - 1
        if last_rec_id in gap_rec_ids:
            gap_rec_ids.remove(last_rec_id)

        # gap ids can only be listed once
        gap_rec_ids = sorted(set(gap_rec_ids))

        # create recording segments from identified gaps
        ncsSects.sects.append(NcsSection(0, ncsMemMap['timestamp'][0], -1, -1, -1))
        for gap_rec_id in gap_rec_ids:
            curr_sec = ncsSects.sects[-1]
            curr_sec.endRec = gap_rec_id
            curr_sec.endTime = pred_times[gap_rec_id]
            n_samples = np.sum(ncsMemMap['nb_valid'][curr_sec.startRec:gap_rec_id + 1])
            curr_sec.n_samples = n_samples

            next_sec = NcsSection(gap_rec_id + 1,
                                  ncsMemMap['timestamp'][gap_rec_id + 1], -1, -1, -1)
            ncsSects.sects.append(next_sec)

        curr_sec = ncsSects.sects[-1]
        curr_sec.endRec = len(ncsMemMap['timestamp']) - 1
        curr_sec.endTime = pred_times[-1]
        n_samples = np.sum(ncsMemMap['nb_valid'][curr_sec.startRec:])
        curr_sec.n_samples = n_samples

        # calculate the estimated frequency of the block with the most samples
        max_blk_idx = np.argmax([bl.endRec - bl.startRec for bl in ncsSects.sects])
        max_blk = ncsSects.sects[max_blk_idx]

        maxBlkFreqEstimate = (max_blk.n_samples - ncsMemMap['nb_valid'][max_blk.endRec]) * 1e6 / \
                             (ncsMemMap['timestamp'][max_blk.endRec] - max_blk.startTime)

        ncsSects.sampFreqUsed = maxBlkFreqEstimate
        ncsSects.microsPerSampUsed = NcsSectionsFactory.get_micros_per_samp_for_freq(
                                                                        maxBlkFreqEstimate)
        # free memory that is unnecessarily occupied by the memmap
        # (see https://github.com/numpy/numpy/issues/19340)
        del ncsMemMap
        return ncsSects

    @staticmethod
    def _buildForMaxGap(ncsMemMap, nomFreq):
        """
        Determine sections of records in memory mapped Ncs file given a nominal frequency of
        the file, using the default values of frequency tolerance and maximum gap between blocks.

        Parameters
        ----------
        ncsMemMap:
            memmap of Ncs file
        nomFreq:
            nominal sampling frequency used, normally from header of file

        Returns
        -------
            NcsSections object
        """
        nb = NcsSections()

        numRecs = ncsMemMap.shape[0]
        if numRecs < 1:
            return nb

        chanNum = ncsMemMap['channel_id'][0]
        ts0 = ncsMemMap['timestamp'][0]

        lastBlkI = numRecs - 1
        lts = ncsMemMap['timestamp'][lastBlkI]
        lcid = ncsMemMap['channel_id'][lastBlkI]
        lnb = ncsMemMap['nb_valid'][lastBlkI]
        lsr = ncsMemMap['sample_rate'][lastBlkI]

        # check if file is one block of records, with exact timestamp match, which may be the case
        numSampsForPred = NcsSection._RECORD_SIZE * lastBlkI
        predLastBlockStartTime = NcsSectionsFactory.calc_sample_time(nomFreq, ts0, numSampsForPred)
        freqInFile = math.floor(nomFreq)
        if lts - predLastBlockStartTime == 0 and lcid == chanNum and lsr == freqInFile:
            endTime = NcsSectionsFactory.calc_sample_time(nomFreq, lts, lnb)
            curBlock = NcsSection(0, ts0, lastBlkI, endTime, numSampsForPred)
            nb.sects.append(curBlock)
            nb.sampFreqUsed = (numSampsForPred + lnb) / (endTime - ts0) * 1e6
            nb.microsPerSampUsed = NcsSectionsFactory.get_micros_per_samp_for_freq(nb.sampFreqUsed)

        # otherwise parse records to determine blocks using default maximum gap length
        else:
            nb.sampFreqUsed = nomFreq
            nb.microsPerSampUsed = NcsSectionsFactory.get_micros_per_samp_for_freq(nb.sampFreqUsed)
            maxGapToAllow = round(NcsSectionsFactory._maxGapSampFrac * 1e6 / nomFreq)
            nb = NcsSectionsFactory._parseForMaxGap(ncsMemMap, nb, maxGapToAllow)

        return nb

    @staticmethod
    def build_for_ncs_file(ncsMemMap, nlxHdr):
        """
        Build an NcsSections object for an NcsFile, given as a memmap and NlxHeader,
        handling gap detection appropriately given the file type as specified by the header.

        Parameters
        ----------
        ncsMemMap:
            memory map of file
        nlxHdr:
            NlxHeader from corresponding file.

        Returns
        -------
        An NcsSections corresponding to the provided ncsMemMap and nlxHdr
        """
        acqType = nlxHdr.type_of_recording()

        # Old Neuralynx style with truncated whole microseconds for actual sampling. This
        # restriction arose from the sampling being based on a master 1 MHz clock.
        if acqType == "PRE4":
            freq = nlxHdr['sampling_rate']
            microsPerSampUsed = math.floor(NcsSectionsFactory.get_micros_per_samp_for_freq(freq))
            sampFreqUsed = NcsSectionsFactory.get_freq_for_micros_per_samp(microsPerSampUsed)
            nb = NcsSectionsFactory._buildGivenActualFrequency(ncsMemMap, sampFreqUsed,
                                                               math.floor(freq))
            nb.sampFreqUsed = sampFreqUsed
            nb.microsPerSampUsed = microsPerSampUsed

        # digital lynx style with fractional frequency and micros per samp determined from
        # block times
        elif acqType == "DIGITALLYNX" or acqType == "DIGITALLYNXSX" or acqType == 'CHEETAH64':
            nomFreq = nlxHdr['sampling_rate']
            nb = NcsSectionsFactory._buildForMaxGap(ncsMemMap, nomFreq)

        # BML & ATLAS style with fractional frequency and micros per samp
        elif acqType == "BML" or acqType == "ATLAS":
            sampFreqUsed = nlxHdr['sampling_rate']
            nb = NcsSectionsFactory._buildGivenActualFrequency(ncsMemMap, sampFreqUsed,
                                                               math.floor(sampFreqUsed))

        else:
            raise TypeError("Unknown Ncs file type from header.")

        return nb

    @staticmethod
    def _verifySectionsStructure(ncsMemMap, ncsSects):
        """
        Check that the record structure and timestamps for the ncsMemMap
        agrees with that in ncsSects.

        Provides a more rapid verification of structure than building a new NcsSections
        and checking equality.

        Parameters
        ----------
        ncsMemMap:
            memmap of file to be checked
        ncsSects
            existing block structure to be checked

        Returns
        -------
            true if all timestamps and block record starts and stops agree, otherwise false.
        """
        for blki in range(0, len(ncsSects.sects)):
            if ncsMemMap['timestamp'][ncsSects.sects[blki].startRec] != \
                    ncsSects.sects[blki].startTime:
                return False
            ets = ncsMemMap['timestamp'][ncsSects.sects[blki].endRec]
            enb = ncsMemMap['nb_valid'][ncsSects.sects[blki].endRec]
            endTime = NcsSectionsFactory.calc_sample_time(ncsSects.sampFreqUsed, ets, enb)
            if endTime != ncsSects.sects[blki].endTime:
                return False

        return True
