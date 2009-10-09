import numpy
import NeuroTools.signals as signals
from NeuroTools.spike2.sonpy import son

"""
Jens Kremkow
INCM-CNRS, Marseille, France
ALUF, Freiburg, Germany
2008
"""


def load(filename,channels=None,start=None,stop=None):
    if channels is None:
        fileheader = son.FileHeader(filename)
        channels = fileheader.return_chan_list()
        
    if len(channels) is 1:
        return load_channel(filename,channel=channels[0],start=None,stop=None)
    
    channel_obj = {}
    for chan in channels:
        try:
            channel_obj[chan] = load_channel(filename,channel=chan,start=None,stop=None)
        except ValueError:
            print 'channel ',chan,' could not be read'
    return channel_obj

def load_channel(filename,channel=1,start=None,stop=None):
    chan = son.Channel(channel,filename)
    data = numpy.array(chan.data(start=start,stop=stop))
    if chan.info.type() == 'Adc':
        dt = chan.info.dt[0]
        obj = Adc(data,dt)
    else:
        exec('obj = %s(data)'%chan.info.type())
    
    obj.blockheader = chan.blockheader
    obj.fhead = chan.fhead
    obj.info = chan.info
    return obj


class Channel(numpy.ndarray):
    """
    Spike2 channels types which are not yet supported by NeuroTools.signals.
    
    timeunits = microseconds,milliseconds,seconds
    """
    def __new__(cls, data):
        return numpy.array(data).view(cls)
    def __init__(self,data):
        self.timeunits = None
    def time(self,timestamps=None,timeunits='seconds'):
        """
        """
        interval = self.info.lChanDvd
        if timeunits== 'ticks':
            interval = 1
        if interval == 0:
            interval = 1.

        if timestamps == None:
            return self.fhead._ticks_to_seconds(numpy.arange(len(self))*interval,timeunits)
        else:
            return self.fhead._ticks_to_seconds(timestamps*interval,timeunits)

    def time_to_ticks(self,timestamps,timeunits='seconds'):
        """
        Converts a timestamp into ticks. The timeunits of the timestamp should be given, default is seconds.
        """
        timestamps = numpy.array(timestamps)
        interval = self.info.lChanDvd[0]
        
        if timeunits is 'microseconds': timestamps = (timestamps/1e6)
        elif timeunits is 'milliseconds': timestamps = (timestamps/1e3)
            
        timestamps = (timestamps/float(interval))/((self.fhead.usPerTime*self.fhead.timePerADC))/self.fhead.dTimeBase
        return timestamps.round().astype(int)
    
    def threshold_detection(self,threshold=None,timeunits='milliseconds',return_events=False,return_SpikeTrain=False):
        """
        
        """
        if threshold == None:
            print 'please give a threhold'
            return 0
        
        above = numpy.where(self > threshold)[0]
        take = (numpy.diff(above))>1.
        take[0] = True
        
        time = self.time(timeunits=timeunits)
        self.events = time[above][take]
        if return_events:
            return self.events
        if return_SpikeTrain:
            if timeunits is not 'milliseconds':
                time = self.time(timeunits='milliseconds')
                events = time[above][take]
                return signals.SpikeTrain(events)
            else:
                return signals.SpikeTrain(self.events)
        
    
    
    


        
class Adc(signals.AnalogSignal):
    """
    Adc represented as analog signal. See NeuroTools.signal.AnalogSignal for further details.
    """
    pass

class EventFall(Channel):
    pass
          
class EventRise(Channel):
    pass
        
class EventBoth(Channel):
    pass       
    
class Marker(Channel):
    def __new__(cls, data):
        
        return numpy.array('times and markers').view(cls)
    def __init__(self,data):
        self.markers = data[:,1]
        self.times = data[:,0].astype('float')
    def get_times(self,marker=None):
        timeunits = 'seconds'
        if markers is None: print 'please give marker'; return None
        
        if timeunits == 'ticks':
            times = self.time_to_ticks(self.times,timeunits=timeunits)
        else:
            times = self.times
            
        return times[self.marker==marker]          

class AdcMark(Channel):
    pass 
    
class RealMark(Channel):
    pass

class TextMark(Channel):
    pass 

class RealWave(Channel):
    pass 

