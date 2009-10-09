import unittest
import os, sys, numpy
sys.path.append(os.path.abspath('../../..'))
from neo.core import SpikeTrain

class SpikeTrainTest(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def testCreateSpikeTrain(self):
        spk = SpikeTrain(numpy.arange(0,110,10))
        assert (spk.t_start == 0) and (spk.t_stop == 100)
        self.assert_( arrays_are_equal(spk.spike_times, numpy.arange(0,110,10)) )
    
    def testCreateSpikeTrainFromList(self):
        spk = SpikeTrain(range(0,110,10))
        assert (spk.t_start == 0) and (spk.t_stop == 100)
        self.assert_( arrays_are_equal(spk.spike_times, numpy.arange(0,110,10)) )
    
    def testCreateSpikeTrainFull(self):
        spk = SpikeTrain(numpy.arange(0,110,10), 0, 100)
        assert (spk.t_start == 0) and (spk.t_stop == 100)
    
    def testCreateWithTStartOnly(self):
        spk = SpikeTrain(numpy.arange(0,110,10), t_start=20)
        assert (spk.t_start == 20) and (spk.t_stop == 100)
        assert arrays_are_equal( spk.spike_times, numpy.arange(20, 110, 10) )
        
    def testCreateWithTStopOnly(self):
        spk = SpikeTrain(numpy.arange(0,110,10), t_stop=70)
        assert (spk.t_start == 0) and (spk.t_stop == 70)
        assert arrays_are_equal( spk.spike_times, numpy.arange(0, 80, 10) )
    
    def testCreateSpikeSmallWrongTimes(self):
        self.assertRaises(Exception, SpikeTrain, numpy.arange(0,110,10), 20, 10)
    
    def testCreateSpikeTrainNegativeTstart(self):
        self.assertRaises(ValueError, SpikeTrain, numpy.arange(0,110,10), -20, 10)
    
    def testCreateSpikeTrainNegativeSpikeTime(self):
        self.assertRaises(ValueError, SpikeTrain, numpy.arange(-100,110,10))
    
    def testCreateWithInvalidValuesInList(self):
        self.assertRaises(ValueError, SpikeTrain, [0.0, "elephant", 0.3, -0.6, 0.15])
    
    def testCopy(self):
        spk = SpikeTrain(numpy.arange(0,110,10), 0, 100)
        spk2 = spk.copy()
        assert spk.is_equal(spk2)
    
    def testDuration(self):
        spk = SpikeTrain(numpy.arange(0,110,10), 0, 100)
        assert spk.duration() == 100
    
    def testMerge(self):
        spk = SpikeTrain(numpy.arange(0,110,10))
        spk2 = SpikeTrain(numpy.arange(100,210,10))
        spk.merge(spk2)
        assert (spk.t_stop == 200) and (len(spk) == 22)
    
    def testTimeAxis(self):
        spk = SpikeTrain(numpy.arange(0,1010,10))
        if newnum:
            assert len(spk.time_axis(100)) == 11
        else:
            assert len(spk.time_axis(100)) == 10
    
    def testAddOffset(self):
        spk = SpikeTrain(numpy.arange(0,1010,10))
        spk.time_offset(50)
        assert (spk.t_start == 50) and (spk.t_stop == 1050) and numpy.all(spk.spike_times == numpy.arange(50,1060,10))
    
    def testTime_Slice(self):
        spk1 = SpikeTrain(numpy.arange(0,1010,10))
        spk1 = spk1.time_slice(250, 750)
        assert len(numpy.extract((spk1.spike_times < 250) | (spk1.spike_times > 750), spk1.spike_times)) == 0
        spk2 = SpikeTrain([0.0, 0.1, 0.3, 0.6, 0.15])
        self.assert_( arrays_are_equal(SpikeTrain([0.15, 0.3]).spike_times,
                                       spk2.time_slice(0.11,0.4).spike_times) ) # should not include 0.1
        self.assert_( arrays_are_equal(SpikeTrain([0.1, 0.15, 0.3]).spike_times,
                                       spk2.time_slice(0.10,0.4).spike_times) ) # should include 0.1
        
    def testIsi(self):
        spk = SpikeTrain(numpy.arange(0,200,10))
        assert numpy.all(spk.isi() == 10)
    
    def testMeanRate(self):
        poisson_param = 1./40
        isi           = numpy.random.exponential(poisson_param, 1000)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk = SpikeTrain(poisson_times)
        assert   35 < spk.mean_rate() < 45
    
    def testMeanRateParams(self):
        poisson_param = 1./40
        isi           = numpy.random.exponential(poisson_param, 1000)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk1 = SpikeTrain(poisson_times,t_start=0,t_stop=5000)
        spk2 = SpikeTrain(range(10), t_stop=10)
        assert   30 < spk1.mean_rate() < 50
        self.assertEqual(spk2.mean_rate(), 1000.0)
        self.assertAlmostEqual(spk2.mean_rate(t_stop=4.99999999999), 1000.0, 6)
        self.assertEqual(spk2.mean_rate(t_stop=5.0), 1200.0)
    
    def testCvIsi(self):
        poisson_param = 1./40
        isi           = numpy.random.exponential(poisson_param, 1000)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk1 = SpikeTrain(poisson_times)
        spk2 = SpikeTrain(range(10), t_stop=10)
        assert 0.9 < spk1.cv_isi() < 1.1
        self.assertEqual(spk2.cv_isi(), 0)


    def testCvKL(self):
        poisson_param = 1./10 # 1 / firing_frequency
        isi           = numpy.random.exponential(poisson_param, 1000)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk1 = SpikeTrain(poisson_times)
        assert 0.9 < spk1.cv_kl(bins = 1000) < 1.1
        # does not depend on bin size
        assert 0.9 < spk1.cv_kl(bins = 100) < 1.1
        # does not depend on time
        poisson_param = 1./4
        isi           = numpy.random.exponential(poisson_param, 1000)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk1 = SpikeTrain(poisson_times)
        assert 0.9 < spk1.cv_kl() < 1.1
        spk2 = SpikeTrain(range(10), t_stop=10)
        self.assertEqual(spk2.cv_isi(), 0)
    
    def testHistogram(self):
        poisson_param = 1./40
        isi           = numpy.random.exponential(poisson_param, 1000)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk = SpikeTrain(poisson_times)
        hist = spk.time_histogram(5000)
        N = len(hist) - 1
        assert numpy.all((30 < hist[0:N]) & (hist[0:N] < 60))
    
    def testVictorPurpuraDistance(self):
        poisson_param = 1./40
        isi           = numpy.random.exponential(poisson_param, 20)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk = SpikeTrain(poisson_times)
        
        isi           = numpy.random.exponential(poisson_param, 20)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk2 = SpikeTrain(poisson_times)
        
        poisson_param = 1./5
        isi           = numpy.random.exponential(poisson_param, 20)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk3 = SpikeTrain(poisson_times)
        
        assert (spk.distance_victorpurpura(spk2,0.1) < spk.distance_victorpurpura(spk3,0.1)) \
                and (spk.distance_victorpurpura(spk, 0.1) == 0)
    
    def testKreuzDistance(self):
        poisson_param = 1./40
        isi           = numpy.random.exponential(poisson_param, 20)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk = SpikeTrain(poisson_times)
        
        isi           = numpy.random.exponential(poisson_param, 20)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk2 = SpikeTrain(poisson_times)
        
        poisson_param = 1./5
        isi           = numpy.random.exponential(poisson_param, 20)
        poisson_times = numpy.cumsum(isi)*1000. # To convert the spikes_time in ms
        spk3 = SpikeTrain(poisson_times)
        
        assert (spk.distance_kreuz(spk2) < spk.distance_kreuz(spk3)) and (spk.distance_kreuz(spk) == 0)

    def testFanoFactorIsi(self):
        spk = SpikeTrain(numpy.arange(0,1010,10))
        assert spk.fano_factor_isi() == 0.
        
if __name__ == "__main__":
    unittest.main()