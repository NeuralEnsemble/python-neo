import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))
from neo.core import SpikeTrainList, SpikeTrain, Neuron
from neo.core.pairgenerator import *

class SpikeTrainListTest(unittest.TestCase):
    
    def setUp(self):
        self.spikes = []
        nb_cells    = 10
        frequencies = nb_cells*[10]
        for idx in xrange(nb_cells):
            param   = 1./frequencies[idx]
            isi     = numpy.random.exponential(param, 1000)
            pspikes = numpy.cumsum(isi)
            neuron  = Neuron(id=idx)
            self.spikes += [SpikeTrain(spike_times=pspikes, neuron=neuron)]            
        self.spk = SpikeTrainList(spiketrains=self.spikes, t_start=0, t_stop=5)
    
    def tearDown(self):
        pass
    
    def testCreateSpikeList(self):
        assert len(self.spk) == 10
    
    def testCreateSpikeList(self):
        assert numpy.all(self.spk.id_list == numpy.arange(10))
    
    def testGetItem(self):
        assert isinstance(self.spk[0], SpikeTrain)
    
    #def testSetItemWrongType(self):
        #self.assertRaises(Exception, self.spk.__setitem__, 0, numpy.arange(100))
    
    #def testSetItem(self):
        #spktrain     = SpikeTrain(spike_times=numpy.arange(10), neuron=Neuron(id=100))
        #self.spk[11] = spktrain
        #assert len(self.spk) == 11
    
    def testGetSlice(self):
        assert len(self.spk[0:5]) == 5
    
    def testAppend(self):
        spktrain = SpikeTrain(spike_times=numpy.arange(10))
        self.assertRaises(Exception, self.spk.append, 0, spktrain)
    
    #def testConcatenate(self):
        #self.assertRaises(Exception, self.spk.concatenate, self.spk)
        
    #def testMerge(self):
        #spk2 = spikes.SpikeList(self.spikes,range(50,60))
        #self.spk.merge(spk2)
        #assert len(self.spk) == 20
            
    def testId_SliceInt(self):
        assert len(self.spk.id_slice(5)) == 5
    
    def testCopy(self):
        spk2 = self.spk.copy()
        assert len(spk2) == len(self.spk) and (spk2[0].is_equal(self.spk[0]))
    
    def testId_SliceList(self):
        assert numpy.all(self.spk.id_slice([0,1,2,3]).id_list== [0,1,2,3])
    
    def testTime_Slice(self):
        spk     = SpikeTrainList(spiketrains=self.spikes, t_start=1)
        new_spk = spk.time_slice(1, 10.)
        assert (new_spk.t_start == spk.t_start) and (new_spk.t_stop == 10.)

    def testAddOffset(self):
        spk2 = self.spk.time_slice(0,100)
        spk2.time_offset(10)
        assert (spk2._t_start == 10) and (spk2._t_stop == 110)

    def testFirstSpikeTime(self):
        assert self.spk.first_spike()[0] >= self.spk._t_start
    
    def testLastSpikeTime(self):
        assert self.spk.last_spike()[0] <= self.spk._t_stop
        
    def testSelect_Ids(self):
        spks     = []
        nb_cells = 3
        frequencies = [5, 100, 100] 
        for idx in xrange(nb_cells):
            param   = 1./frequencies[idx]
            isi     = numpy.random.exponential(param, 1000)
            pspikes = numpy.cumsum(isi)
            neuron  = Neuron(id=idx)
            spks   += [SpikeTrain(spike_times=pspikes, neuron=neuron)]
        spk = SpikeTrainList(spiketrains=spks, t_start=0, t_stop=1)
        assert len(spk.select_ids("cell.mean_rate() < 20")) == 1

    #def testIsis(self):
        #pass
    
    def testCV_Isis(self):
        assert 0.8 < numpy.mean(self.spk.cv_isi()) < 1.2
    
    #def testCVKL(self):
        #assert 0.8 < numpy.mean(self.spk.cv_kl()) < 1.2
        
    #def testCVLocal(self):
        #assert 0.8 < self.spk.cv_local() < 1.2
    
    def testMeanRate(self):
        assert 5 < self.spk.mean_rate() < 15
    
    def testMeanRates(self):
        correct = True
        rates = self.spk.mean_rates()
        for idx in xrange(len(self.spk.id_list)):
            if not(5 < rates[idx] < 15):
                correct = False
        assert correct
    
    #def testMeanRateStd(self):
        #assert self.spk.mean_rate_std() >= 0

    #def testMeanRateVarianceAndCovariance(self):
        #assert (abs(self.spk.mean_rate_variance(10) - self.spk.mean_rate_covariance(self.spk, 10)) < 0.01)

    #def testPairwise_Pearson_CorrCoeff(self):
        #x1,y1 = self.spk.pairwise_pearson_corrcoeff(10, time_bin=0.001)
        #assert x1 < y1

    def testVictorPurpuraDistance(self):
        d_spike = self.spk.distance_victorpurpura(10, cost=0.1)
        d_rate  = self.spk.distance_victorpurpura(10, cost=0.9)
        assert (d_rate > d_spike)
    
    def testKreuzDistance(self):
        d_self = self.spk.distance_kreuz(10)
        assert d_self > 0
    
    def testCrossCorrZero(self):
        cc1 = self.spk.pairwise_cc_zero(5, AutoPairs(self.spk, self.spk), time_bin=0.1)
        cc2 = self.spk.pairwise_cc_zero(5, RandomPairs(self.spk, self.spk), time_bin=0.1)
        assert (0 <= cc1 <= 1) and (cc1 > cc2)

    def testFanoFactor(self):
        assert 0.9 < self.spk.fano_factor(0.005) < 1.1

    #def testIdOffset(self):
        #self.spk.id_offset(100)
        #assert numpy.all(self.spk.id_list == numpy.arange(100,110))

    def testPairwiseCCZero(self):
        self.spk.pairwise_cc_zero(50, bin_size=0.01, time_window=100)
        self.spk.pairwise_cc_zero(50, RandomPairs(self.spk, self.spk), bin_size=10., time_window=200)
        self.spk.pairwise_cc_zero(50, AutoPairs(self.spk, self.spk), bin_size=10., time_window=200)
   
    
    
if __name__ == "__main__":
    unittest.main()