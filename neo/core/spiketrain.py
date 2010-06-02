# -*- coding: utf-8 -*-
import numpy
from neuron import Neuron

class SpikeTrain(object):
     
    """
    A group of :class:`Spike` (or spike times) emitted by the same :class:`Neuron`

    **Definition**
    A :class:`SpikeTrain` is an array of :class:`Spike` emitted by the same
    :class:`Neuron`. The spike times of the :class:`SpikeTrain` can be viewed as 
    an array of :class:`Spike` object, with times and waveforms, or just as an array
    of times.

    with arguments:
    
    ``spikes`` or ``spike_times``
        - **spikes**       : an array/list of :class:`Spike` 
        - **spike_times ** : an array/list of times

    ``neuron``
        The :class:`Neuron` emitting the :class:`SpikeTrain`

    ``t_start``
        The absolute beginning (in second) of the :class:`SpikeTrain`
    
    ``t_stop``
        The absolute end (in second) of the :class:`SpikeTrain`    

    **Usage**    
    
    **Example**

    """
    
    
    def __init__(self, *arg, **karg):
                        
        self._spike_times = None
        self._spikes      = [ ]
        self._t_start     = None
        self._t_stop      = None
        self.neuron       = None
        
        if karg.has_key('spike_times'):
            self._spike_times = numpy.array(karg['spike_times'])
            self._spike_times.sort()
            
        #~ if karg.has_key('t_start'):
            #~ self._t_start = karg['t_start']
        #~ if karg.has_key('t_stop'):
            #~ self._t_stop = karg['t_stop']
        #~ if karg.has_key('interval'):
            #~ self._t_start, self._t_stop = karg['interval']

        # May be useful to check the times. But should be adapted for spike object instead
        # of spike times in self.spike_times
        #~ self.__calc_startstop()

        if karg.has_key('neuron'):
            self.neuron = karg['neuron']
        else:
            self.neuron = None
        
        if karg.has_key('spikes'):
            self._spikes = karg['spikes']
            
    @property
    def spike_times(self):
        if self._spike_times is None:
            self._spike_times = numpy.empty(len(self._spikes))
            #~ print len(self._spikes), self._spikes[0]
            for count, sp in enumerate(self._spikes):
                self._spike_times[count] = sp.time
            return self._spike_times
        else:
            return self._spike_times
    
    @property
    def spikes(self):
        return self._spikes
    
    #~ @property
    #~ def t_start(self):
        #~ return self._t_start
    
    #~ @property
    #~ def t_stop(self):
        #~ return self._t_stop
    
    def get_spikes(self):
        return self._spikes
        
    def __calc_startstop(self):
        size = len(self)
        if size == 0:
            if self._t_start is None: 
                self._t_start = 0
            if self._t_stop is None:
                self._t_stop  = 0.1
        elif size == 1: # spike list may be empty
            if self._t_start is None:
                self._t_start = self.spike_times[0]
            if self._t_stop is None:
                self._t_stop = self.spike_times[0] + 0.1
        elif size > 1:
            if self._t_start is None:
                self._t_start = self.spike_times.min()
            if self._t_stop is None:
                self._t_stop = self.spike_times.max()
            
        if self._t_start >= self._t_stop :
            raise Exception("Incompatible time interval : t_start = %s, t_stop = %s" % (self._t_start, self._t_stop))
        if self._t_start < 0:
            raise ValueError("t_start must not be negative")
        if numpy.any(self.spike_times < 0):
            raise ValueError("Spike times must not be negative")
    
    
    #~ def __str__(self):
        #~ res = "SpikeTrain"
        #~ if self.neuron:
            #~ res += " emitted by neuron %s" % str(self.neuron)
        #~ res += " has %d spikes:\n %s" %(len(self), str(self.spike_times))
        #~ return res

    def __iter__(self):
        if self._spike_times is not None:
            return iter(self._spike_times)
        else:
            return iter(self._spikes)

    #~ def __len__(self):
        #~ print self._spike_times
        #~ if self._spike_times is not None:
            #~ return len(self._spike_times)
        #~ else:
            #~ return len(self._spikes)   

    def duration(self):
        """
        Return the duration of the SpikeTrain
        """
        return self._t_stop - self._t_start
    
    def time_parameters(self):
        """
        Return the time parameters of the SpikeTrain (t_start, t_stop)
        """
        return (self.t_start, self.t_stop)
    
    def is_equal(self, spktrain):
        """
        Return True if the SpikeTrain object is equal to one other SpikeTrain, i.e
        if they have same time parameters and same spikes_times
        
        Inputs:
            spktrain - A SpikeTrain object
        
        See also:
            time_parameters()
        """
        test = (self.time_parameters() == spktrain.time_parameters())
        return numpy.all(self.spike_times == spktrain.spike_times) and test
    
    def copy(self):
        """
        Return a copy of the SpikeTrain object
        """
        return SpikeTrain(spike_times=self.spike_times, t_start=self._t_start, t_stop=self._t_stop)

    
    def merge(self, spiketrain):
        """
        Add the spike times from a spiketrain to the current SpikeTrain
        
        Inputs:
            spiketrain - The SpikeTrain that should be added
        
        Examples:
            >> a = SpikeTrain(range(0,100,10),0.1,0,100)
            >> b = SpikeTrain(range(400,500,10),0.1,400,500)
            >> a.merge(b)
            >> a.spike_times
                [   0.,   10.,   20.,   30.,   40.,   50.,   60.,   70.,   80.,
                90.,  400.,  410.,  420.,  430.,  440.,  450.,  460.,  470.,
                480.,  490.]
            >> a.t_stop
                500
        """
        self._spike_times = numpy.insert(self.spike_times, self.spike_times.searchsorted(spiketrain.spike_times), \
                                        spiketrain.spike_times)
        self._t_start     = min(self._t_start, spiketrain._t_start)
        self._t_stop      = max(self._t_stop, spiketrain._t_stop)


#################### INTERNAL STRUCTURE SHOULD BE CHANGED#######################

    def time_offset(self, offset):
        """
        Add an offset to the SpikeTrain object. t_start and t_stop are
        shifted from offset, so does all the spike times.
         
        Inputs:
            offset - the time offset, in ms
        
        Examples:
            >> spktrain = SpikeTrain(arange(0,100,10))
            >> spktrain.time_offset(50)
            >> spklist.spike_times
                [  50.,   60.,   70.,   80.,   90.,  100.,  110.,  
                120.,  130.,  140.]
        """
        self._t_start += offset
        self._t_stop  += offset
        if self._spikes is not None:
            for spike in self._spikes:
                spike.time += offset
        else:
            self._spike_times += offset

    def jitter(self, jitter):
        """
        Returns the SpikeTrain with spiketimes jittered by a normal distribution.

        Inputs:
              jitter - sigma of the normal distribution

        Examples:
              >> st_jittered = st.jitter(2.0)
        """
        
        noise = jitter * numpy.random.normal(loc=0.0, scale=1.0, size=len(self.spike_times))
        
        if self._spikes is not None:
            for count, spike in enumerate(self._spikes):
                spike.time += noise[count]
        else:
            self._spike_times += noise

        ###### Should we adapt t_start and t_stop in consequence ???? ###########




#################### SHARED OPERATIONS #######################

    def isi(self):
        """
        Return an array with the inter-spike intervals of the SpikeTrain
        
        Examples:
            >> st.spikes_times=[0, 2.1, 3.1, 4.4]
            >> st.isi()
                [2.1, 1., 1.3]
        
        See also
            cv_isi
        """
        return numpy.diff(self.spike_times)

    def mean_rate(self, t_start=None, t_stop=None):
        """ 
        Returns the mean firing rate between t_start and t_stop, in Hz
        
        Inputs:
            t_start - in ms. If not defined, the one of the SpikeTrain object is used
            t_stop  - in ms. If not defined, the one of the SpikeTrain object is used
        
        Examples:
            >> spk.mean_rate()
                34.2
        """
        if t_start is None: 
            t_start = self._t_start
        if t_stop is None: 
            t_stop=self._t_stop
        idx = numpy.where((self._spike_times >= t_start) & (self._spike_times <= t_stop))[0]
        return len(idx)/(t_stop-t_start)

    def cv_isi(self):
        """
        Return the coefficient of variation of the isis.
        
        cv_isi is the ratio between the standard deviation and the mean of the ISI
          The irregularity of individual spike trains is measured by the squared
        coefficient of variation of the corresponding inter-spike interval (ISI)
        distribution normalized by the square of its mean.
          In point processes, low values reflect more regular spiking, a
        clock-like pattern yields CV2= 0. On the other hand, CV2 = 1 indicates
        Poisson-type behavior. As a measure for irregularity in the network one
        can use the average irregularity across all neurons.
        
        http://en.wikipedia.org/wiki/Coefficient_of_variation
        
        See also
            isi, cv_kl
            
        """
        isi = self.isi()
        if len(isi) > 0:
            return numpy.std(isi)/numpy.mean(isi)
        else:
            logging.debug("Warning, a CV can't be computed because there are not enough spikes")
            return numpy.nan

    def cv_kl(self, bins=100):
        """
        Provides a measure for the coefficient of variation to describe the
        regularity in spiking networks. It is based on the Kullback-Leibler
        divergence and decribes the difference between a given
        interspike-interval-distribution and an exponential one (representing
        poissonian spike trains) with equal mean.
        It yields 1 for poissonian spike trains and 0 for regular ones.
        
        Reference:
            http://incm.cnrs-mrs.fr/LaurentPerrinet/Publications/Voges08fens
        
        Inputs:
            bins - the number of bins used to gather the ISI
        
        Examples:
            >> spklist.cv_kl(100)
                0.98
        
        See also:
            cv_isi
            
        """
        isi = self.isi()
        if len(isi) < 2:
            logging.debug("Warning, a CV can't be computed because there are not enough spikes")
            return numpy.nan
        else:
            proba_isi, xaxis = numpy.histogram(isi, bins=bins, normed=True, new=True)
            proba_isi /= numpy.sum(proba_isi)
            bin_size   = xaxis[1]-xaxis[0]
            # differential entropy: http://en.wikipedia.org/wiki/Differential_entropy
            KL   = - numpy.sum(proba_isi * numpy.log(proba_isi+1e-16)) + numpy.log(bin_size)
            KL  -= -numpy.log(self.mean_rate()) + 1.
            CVkl =  numpy.exp(-KL)
            return CVkl
    

    def fano_factor_isi(self):
        """ 
        Return the fano factor of this spike trains ISI.
        
        The Fano Factor is defined as the variance of the isi divided by the mean of the isi
        
        http://en.wikipedia.org/wiki/Fano_factor
        
        See also
            isi, cv_isi
        """
        isi = self.isi()
        if len(isi) > 0:
            fano = numpy.var(isi)/numpy.mean(isi)
            return fano
        else: 
            raise Exception("No spikes in the SpikeTrain !")

    def time_slice(self, t_start, t_stop):
        """ 
        Return a new SpikeTrain obtained by slicing between t_start and t_stop. The new 
        t_start and t_stop values of the returned SpikeTrain are the one given as arguments
        
        Inputs:
            t_start - begining of the new SpikeTrain, in ms.
            t_stop  - end of the new SpikeTrain, in ms.

        Examples:
            >> spk = spktrain.time_slice(0,100)
            >> spk.t_start
                0
            >> spk.t_stop
                100
        """
        idx = numpy.where((self._spike_times >= t_start) & (self._spike_times <= t_stop))[0]
        if self.spikes:        
            return SpikeTrain(spikes=self._spikes[idx], t_start=t_start, t_stop=t_stop, neuron=self.neuron)        
        else:
            return SpikeTrain(spike_times = self._spike_times[idx], t_start=t_start, t_stop=t_stop, neuron=self.neuron)        

    def set_times(self, t_start, t_stop):
        """ 
        Adapt the times (start/stop) of the SpikeTrain obtained by slicing between t_start and t_stop. 
        The t_start and t_stop values of the SpikeTrain are now the one given as arguments, and 
        all event out of this time interval are discarded.
        
        Inputs:
            t_start - begining of the SpikeTrain, in ms.
            t_stop  - end of the SpikeTrain, in ms.

        Examples:
            >> spk = spktrain.time_slice(0,100)
            >> spk.t_start
                0
            >> spk.t_stop
                100
        """
        idx = numpy.where((self._spike_times >= t_start) & (self._spike_times <= t_stop))[0]
        self._t_start = t_start
        self._t_stop  = t_stop
        if self.spikes:        
            self._spikes  = self._spikes[idx]
        self._spike_times = self._spike_times[idx]


    def time_histogram(self, t_start=None, t_stop=None, bin_size=0.01, normalized=True):
        """
        Bin the spikes with the specified bin width. The first and last bins
        are calculated from `self.t_start` and `self.t_stop`.
        
        Inputs:
            bin_size   - the bin size (in s) for gathering spikes_times
            
            normalized - if True, the bin values are scaled to represent firing rates
                         in spikes/second, otherwise otherwise it's the number of spikes 
                         per bin.
        
        Examples:
            >> st=SpikeTrain(range(0,100,5),0.1,0,100)
            >> st.time_histogram(10)
                [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
            >> st.time_histogram(10, normalized=False)
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        
        See also
            time_axis
        """
        ### bins = self.time_axis(bin_size)
        if t_start is None:
            t_start = self._t_start
        if t_stop is None:
            t_stop = self.t_stop
        bins = numpy.arange(t_start, t_stop, bin_size)
        hist, edges = numpy.histogram(self.spike_times, bins)
        if normalized: # what about normalization if time_bin is a sequence?
            hist *= 1/bin_size
        return hist, edges

    
    def distance_victorpurpura(self, spktrain, cost=0.5):
        """
        Function to calculate the Victor-Purpura distance between two spike trains.
        See J. D. Victor and K. P. Purpura,
            Nature and precision of temporal coding in visual cortex: a metric-space
            analysis.,
            J Neurophysiol,76(2):1310-1326, 1996
        
        Inputs:
            spktrain - the other SpikeTrain
            cost     - The cost parameter. See the paper for more information
        """
        nspk_1      = len(self)
        nspk_2      = len(spktrain)
        if cost == 0: 
            return abs(nspk_1-nspk_2)
        elif cost > 1e9 :
            return nspk_1+nspk_2
        scr      = numpy.zeros((nspk_1+1,nspk_2+1))
        scr[:,0] = numpy.arange(0,nspk_1+1)
        scr[0,:] = numpy.arange(0,nspk_2+1)
            
        if nspk_1 > 0 and nspk_2 > 0:
            for i in xrange(1, nspk_1+1):
                for j in xrange(1, nspk_2+1):
                    scr[i,j] = min(scr[i-1,j]+1,scr[i,j-1]+1)
                    scr[i,j] = min(scr[i,j],scr[i-1,j-1]+cost*abs(self.spike_times[i-1]-spktrain.spike_times[j-1]))
        return scr[nspk_1,nspk_2]


    def distance_kreuz(self, spktrain, dt=0.001):
        """
        Function to calculate the Kreuz/Politi distance between two spike trains
        See  Kreuz, T.; Haas, J.S.; Morelli, A.; Abarbanel, H.D.I. & Politi, A. 
            Measuring spike train synchrony. 
            J Neurosci Methods, 165:151-161, 2007

        Inputs:
            spktrain - the other SpikeTrain
            dt       - the bin width used to discretize the spike times
        
        Examples:
            >> spktrain.KreuzDistance(spktrain2)
        
        See also
            VictorPurpuraDistance
        """
        N              = (self.t_stop-self.t_start)/dt
        vec_1          = numpy.zeros(N, numpy.float32)
        vec_2          = numpy.zeros(N, numpy.float32)
        result         = numpy.zeros(N, float)
        idx_spikes     = numpy.array(self.spike_times/dt,int)
        previous_spike = 0
        if len(idx_spikes) > 0:
            for spike in idx_spikes[1:]:
                vec_1[previous_spike:spike] = (spike-previous_spike)
                previous_spike = spike
        idx_spikes     = numpy.array(spktrain.spike_times/dt,int)
        previous_spike = 0
        if len(idx_spikes) > 0:
            for spike in idx_spikes[1:]:
                vec_2[previous_spike:spike] = (spike-previous_spike)
                previous_spike = spike
        idx = numpy.where(vec_1 < vec_2)[0]
        result[idx] = vec_1[idx]/vec_2[idx] - 1
        idx = numpy.where(vec_1 > vec_2)[0]
        result[idx] = -vec_2[idx]/vec_1[idx] + 1
        return numpy.sum(numpy.abs(result))/len(result)
    
    def cross_correlate(self, spiketrain, bin_size=0.001, t_before=0.05, t_after=0.05):
        return self.psth(spiketrain)


    def all_diff_combinate(self, events):
        """
        Return  a vector that combinate all diff between spike times and a list of event.
        """
        t1 = events
        t2 = self.spike_times
        m1 = numpy.tile(t1[:,numpy.newaxis] , (1,t2.size) )
        m2 = numpy.tile(t2[numpy.newaxis,:] , (t1.size,1) )
        m = m2-m1
        m = m.reshape(m.size)        
        return m

    def psth(self, events, bin_size=0.001, t_before=0.05, t_after=0.05):
        """
        Return the psth of the spike times contained in the SpikeTrain according to selected events, 
        on a time window t_spikes - tmin, t_spikes + tmax
        
        Inputs:
            events  - List of Even objects (and events can be the spikes) or just a list 
                      of times
            bin_size- The bin_size bin (in second) used to gather the spike for the psth
            t_min   - Time (>0) to average the signal before an event, in ms (default 0)
            t_max   - Time (>0) to average the signal after an event, in ms  (default 100)
            
        Examples: 
            >> spk.psth(range(0,1000,10), display=True)
            
        See also
            SpikeTrain.spike_histogram
        """
        
        #if isinstance(events, SpikeTrain):
            #events = events.spike_times
        #assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"
        #assert len(events) > 0, "events should not be empty and should contained at least one element"

        # patch by S garcia to be discuss
        
        m = self.all_diff_combinate(events)
        y,x = numpy.histogram( m , bins = numpy.arange(-t_before, t_after, bin_size))
        return y.astype('f')/y.size


        # pierre yger:
        
        #~ spk_hist = self.time_histogram(bin_size)
        #~ subplot  = get_display(display)
        #~ count    = 0
        #~ t_min_l  = numpy.floor(t_min/bin_size)
        #~ t_max_l  = numpy.floor(t_max/bin_size)
        #~ result   = numpy.zeros((t_min_l+t_max_l), numpy.float32)
        #~ t_start  = numpy.floor(self._t_start/bin_size)
        #~ t_stop   = numpy.floor(self._t_stop/bin_size)
        #~ for ev in events:
           #~ ev = numpy.floor(ev/bin_size)
           #~ if ((ev - t_min_l )> t_start) and (ev + t_max_l ) < t_stop:
               #~ count  += 1
               #~ result += spk_hist[(ev-t_min_l):ev+t_max_l]
        #~ result /= count

        #~ return result
    