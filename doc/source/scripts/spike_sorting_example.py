"""
Example for usecases.rst
"""

import numpy as np
from neo import Segment, AnalogSignal, SpikeTrain, Group, ChannelView
from quantities import Hz

# generate some fake data
seg = Segment()
seg.analogsignals.append(
    AnalogSignal(
        [
            [0.1, 0.1, 0.1, 0.1],
            [-2.0, -2.0, -2.0, -2.0],
            [0.1, 0.1, 0.1, 0.1],
            [-0.1, -0.1, -0.1, -0.1],
            [-0.1, -0.1, -0.1, -0.1],
            [-3.0, -3.0, -3.0, -3.0],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
        ],
        sampling_rate=1000 * Hz,
        units="V",
    )
)

# extract spike trains from all channels
st_list = []
for signal in seg.analogsignals:
    # use a simple threshhold detector
    spike_mask = np.where(np.min(signal.magnitude, axis=1) < -1.0)[0]

    # create a spike train
    spike_times = signal.times[spike_mask]
    st = SpikeTrain(spike_times, t_start=signal.t_start, t_stop=signal.t_stop)

    # remember the spike waveforms
    wf_list = []
    for spike_idx in np.nonzero(spike_mask)[0]:
        wf_list.append(signal[spike_idx - 1 : spike_idx + 2, :])
    st.waveforms = np.array(wf_list)

    st_list.append(st)

unit = Group()
unit.spiketrains = st_list
unit.analogsignals.extend(seg.analogsignals)
