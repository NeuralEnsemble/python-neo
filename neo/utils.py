# TODO: proxy objects
# TODO: unittests
# TODO: do we want/need this header?
"""
Convenience functions to extend the functionality of the Neo framework
version 0.5.

Authors: Julia Sprenger, Lyuba Zehl, Michael Denker


Copyright (c) 2017, Institute of Neuroscience and Medicine (INM-6),
Forschungszentrum Juelich, Germany
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import neo
import copy
import warnings
import inspect
import numpy as np
import quantities as pq


def get_events(container, properties=None):
    """
    This function returns a list of Neo Event objects, corresponding to given
    key-value pairs in the attributes or annotations of the Event.

    Parameter:
    ---------
    container: neo.Block or neo.Segment
        The Neo Block or Segment object to extract data from.
    properties: dictionary
        A dictionary that contains the Event keys and values to filter for.
        Each key of the dictionary is matched to a attribute or an an
        annotation of Event. The value of each dictionary entry corresponds to
        a valid entry or a list of valid entries of the attribute or
        annotation.

        If the value belonging to the key is a list of entries of the same
        length as the number of events in the Event object, the list entries
        are matched to the events in the Event object. The resulting Event
        object contains only those events where the values match up.

        Otherwise, the value is compared to the attributes or annotation of the
        Event object as such, and depending on the comparison, either the
        complete Event object is returned or not.

        If None or an empty dictionary is passed, all Event Objects will be
        returned in a list.

    Returns:
    --------
    events: list
        A list of Event objects matching the given criteria.

    Example:
    --------
        >>> event = neo.Event(
                times = [0.5, 10.0, 25.2] * pq.s)
        >>> event.annotate(
                event_type = 'trial start',
                trial_id = [1, 2, 3])
        >>> seg = neo.Segment()
        >>> seg.events = [event]

        # Will return a list with the complete event object
        >>> get_events(seg, properties={'event_type': 'trial start'})

        # Will return an empty list
        >>> get_events(seg, properties={'event_type': 'trial stop'})

        # Will return a list with an Event object, but only with trial 2
        >>> get_events(seg, properties={'trial_id': 2})

        # Will return a list with an Event object, but only with trials 1 and 2
        >>> get_events(seg, properties={'trial_id': [1, 2]})
    """
    if isinstance(container, neo.Segment):
        return _get_from_list(container.events, prop=properties)

    elif isinstance(container, neo.Block):
        event_lst = []
        for seg in container.segments:
            event_lst += _get_from_list(seg.events, prop=properties)
        return event_lst
    else:
        raise TypeError(
            'Container needs to be of type neo.Block or neo.Segment, not %s '
            'in order to extract Events.' % (type(container)))


def get_epochs(container, properties=None):
    """
    This function returns a list of Neo Epoch objects, corresponding to given
    key-value pairs in the attributes or annotations of the Epoch.

    Parameters:
    -----------
    container: neo.Block or neo.Segment
        The Neo Block or Segment object to extract data from.
    properties: dictionary
        A dictionary that contains the Epoch keys and values to filter for.
        Each key of the dictionary is matched to an attribute or an an
        annotation of the Event. The value of each dictionary entry corresponds
        to a valid entry or a list of valid entries of the attribute or
        annotation.

        If the value belonging to the key is a list of entries of the same
        length as the number of epochs in the Epoch object, the list entries
        are matched to the epochs in the Epoch object. The resulting Epoch
        object contains only those epochs where the values match up.

        Otherwise, the value is compared to the attribute or annotation of the
        Epoch object as such, and depending on the comparison, either the
        complete Epoch object is returned or not.

        If None or an empty dictionary is passed, all Epoch Objects will
        be returned in a list.

    Returns:
    --------
    epochs: list
        A list of Epoch objects matching the given criteria.

    Example:
    --------
        >>> epoch = neo.Epoch(
                times = [0.5, 10.0, 25.2] * pq.s,
                durations = [100, 100, 100] * pq.ms)
        >>> epoch.annotate(
                event_type = 'complete trial',
                trial_id = [1, 2, 3]
        >>> seg = neo.Segment()
        >>> seg.epochs = [epoch]

        # Will return a list with the complete event object
        >>> get_epochs(seg, prop={'epoch_type': 'complete trial'})

        # Will return an empty list
        >>> get_epochs(seg, prop={'epoch_type': 'error trial'})

        # Will return a list with an Event object, but only with trial 2
        >>> get_epochs(seg, prop={'trial_id': 2})

        # Will return a list with an Event object, but only with trials 1 and 2
        >>> get_epochs(seg, prop={'trial_id': [1, 2]})
    """
    if isinstance(container, neo.Segment):
        return _get_from_list(container.epochs, prop=properties)

    elif isinstance(container, neo.Block):
        epoch_list = []
        for seg in container.segments:
            epoch_list += _get_from_list(seg.epochs, prop=properties)
        return epoch_list
    else:
        raise TypeError(
            'Container needs to be of type neo.Block or neo.Segment, not %s '
            'in order to extract Epochs.' % (type(container)))


def _get_from_list(input_list, prop=None):
    """
    Internal function
    """
    output_list = []
    # empty or no dictionary
    if not prop or bool([b for b in prop.values() if b == []]):
        output_list += [e for e in input_list]
    # dictionary is given
    else:
        for ep in input_list:
            sparse_ep = ep.copy()
            for k in prop.keys():
                sparse_ep = _filter_event_epoch(sparse_ep, k, prop[k])
                # if there is nothing left, it cannot filtered
                if sparse_ep is None:
                    break
            if sparse_ep is not None:
                output_list.append(sparse_ep)
    return output_list


def _filter_event_epoch(obj, annotation_key, annotation_value):
    """
    Internal function.

    This function returns a copy of a neo Event or Epoch object, which only
    contains attributes or annotations corresponding to requested key-value
    pairs.

    Parameters:
    -----------
    obj : neo.Event
        The neo Event or Epoch object to modify.
    annotation_key : string, int or float
        The name of the annotation used to filter.
    annotation_value : string, int, float, list or np.ndarray
        The accepted value or list of accepted values of the attributes or
        annotations specified by annotation_key. For each entry in obj the
        respective annotation defined by annotation_key is compared to the
        annotation value. The entry of obj is kept if the attribute or
        annotation is equal or contained in annotation_value.

    Returns:
    --------
    obj : neo.Event or neo.Epoch
        The Event or Epoch object with every event or epoch removed that does
        not match the filter criteria (i.e., where none of the entries in
        annotation_value match the attribute or annotation annotation_key.
    """
    valid_ids = _get_valid_ids(obj, annotation_key, annotation_value)

    if len(valid_ids) == 0:
        return None

    return _event_epoch_slice_by_valid_ids(obj, valid_ids)


def _event_epoch_slice_by_valid_ids(obj, valid_ids):
    """
    Internal function
    """
    # modify annotations
    sparse_annotations = _get_valid_annotations(obj, valid_ids)

    # modify array annotations
    sparse_array_annotations = {key: value[valid_ids]
                                for key, value in obj.array_annotations.items() if len(value)}

    if type(obj) is neo.Event:
        sparse_obj = neo.Event(
            times=copy.deepcopy(obj.times[valid_ids]),
            units=copy.deepcopy(obj.units),
            name=copy.deepcopy(obj.name),
            description=copy.deepcopy(obj.description),
            file_origin=copy.deepcopy(obj.file_origin),
            array_annotations=sparse_array_annotations,
            **sparse_annotations)
    elif type(obj) is neo.Epoch:
        sparse_obj = neo.Epoch(
            times=copy.deepcopy(obj.times[valid_ids]),
            durations=copy.deepcopy(obj.durations[valid_ids]),
            units=copy.deepcopy(obj.units),
            name=copy.deepcopy(obj.name),
            description=copy.deepcopy(obj.description),
            file_origin=copy.deepcopy(obj.file_origin),
            array_annotations=sparse_array_annotations,
            **sparse_annotations)
    else:
        raise TypeError('Can only slice Event and Epoch objects by valid IDs.')

    return sparse_obj


def _get_valid_ids(obj, annotation_key, annotation_value):
    """
    Internal function
    """
    # wrap annotation value to be list
    if not type(annotation_value) in [list, np.ndarray]:
        annotation_value = [annotation_value]

    # get all real attributes of object
    attributes = inspect.getmembers(obj)
    attributes_names = [t[0] for t in attributes if not(
        t[0].startswith('__') and t[0].endswith('__'))]
    attributes_ids = [i for i, t in enumerate(attributes) if not(
        t[0].startswith('__') and t[0].endswith('__'))]

    # check if annotation is present
    value_avail = False
    if annotation_key in obj.annotations:
        check_value = obj.annotations[annotation_key]
        value_avail = True
    elif annotation_key in obj.array_annotations:
        check_value = obj.array_annotations[annotation_key]
        value_avail = True
    elif annotation_key in attributes_names:
        check_value = attributes[attributes_ids[
            attributes_names.index(annotation_key)]][1]
        value_avail = True

    if value_avail:
        # check if annotation is list and fits to length of object list
        if not _is_annotation_list(check_value, len(obj)):
            # check if annotation is single value and fits to requested value
            if check_value in annotation_value:
                valid_mask = np.ones(obj.shape)
            else:
                valid_mask = np.zeros(obj.shape)
                if type(check_value) != str:
                    warnings.warn(
                        'Length of annotation "%s" (%s) does not fit '
                        'to length of object list (%s)' % (
                            annotation_key, len(check_value), len(obj)))

        # extract object entries, which match requested annotation
        else:
            valid_mask = np.zeros(obj.shape)
            for obj_id in range(len(obj)):
                if check_value[obj_id] in annotation_value:
                    valid_mask[obj_id] = True
    else:
        valid_mask = np.zeros(obj.shape)

    valid_ids = np.where(valid_mask)[0]

    return valid_ids


def _get_valid_annotations(obj, valid_ids):
    """
    Internal function
    """
    sparse_annotations = copy.deepcopy(obj.annotations)
    for key in sparse_annotations:
        if _is_annotation_list(sparse_annotations[key], len(obj)):
            sparse_annotations[key] = list(np.array(sparse_annotations[key])[
                valid_ids])
    return sparse_annotations


def _is_annotation_list(value, exp_length):
    """
    Internal function
    """
    return (
        (isinstance(value, list) or (
            isinstance(value, np.ndarray) and value.ndim > 0)) and
        (len(value) == exp_length))


def add_epoch(
        segment, event1, event2=None, pre=0 * pq.s, post=0 * pq.s,
        attach_result=True, **kwargs):
    """
    Create epochs around a single event, or between pairs of events. Starting
    and end time of the epoch can be modified using pre and post as offsets
    before the and after the event(s). Additional keywords will be directly
    forwarded to the epoch intialization.

    Parameters:
    -----------
    sgement : neo.Segment
        The segment in which the final Epoch object is added.
    event1 : neo.Event
        The Neo Event objects containing the start events of the epochs. If no
        event2 is specified, these event1 also specifies the stop events, i.e.,
        the epoch is cut around event1 times.
    event2: neo.Event
        The Neo Event objects containing the stop events of the epochs. If no
        event2 is specified, event1 specifies the stop events, i.e., the epoch
        is cut around event1 times. The number of events in event2 must match
        that of event1.
    pre, post: Quantity (time)
        Time offsets to modify the start (pre) and end (post) of the resulting
        epoch. Example: pre=-10*ms and post=+25*ms will cut from 10 ms before
        event1 times to 25 ms after event2 times
    attach_result: bool
        If True, the resulting Neo Epoch object is added to segment.

    Keyword Arguments:
    ------------------
    Passed to the Neo Epoch object.

    Returns:
    --------
    epoch: neo.Epoch
        An Epoch object with the calculated epochs (one per entry in event1).
    """
    if event2 is None:
        event2 = event1

    if not isinstance(segment, neo.Segment):
        raise TypeError(
            'Segment has to be of type neo.Segment, not %s' % type(segment))

    for event in [event1, event2]:
        if not isinstance(event, neo.Event):
            raise TypeError(
                'Events have to be of type neo.Event, not %s' % type(event))

    if len(event1) != len(event2):
        raise ValueError(
            'event1 and event2 have to have the same number of entries in '
            'order to create epochs between pairs of entries. Match your '
            'events before generating epochs. Current event lengths '
            'are %i and %i' % (len(event1), len(event2)))

    times = event1.times + pre
    durations = event2.times + post - times

    if any(durations < 0):
        raise ValueError(
            'Can not create epoch with negative duration. '
            'Requested durations %s.' % durations)
    elif any(durations == 0):
        raise ValueError('Can not create epoch with zero duration.')

    if 'name' not in kwargs:
        kwargs['name'] = 'epoch'
    if 'labels' not in kwargs:
        # this needs to be changed to '%s_%i' % (kwargs['name'], i) for i in range(len(times))]
        # when labels become unicode
        kwargs['labels'] = [
            b'%s_%i' % (kwargs['name'].encode('ascii'), i) for i in range(len(times))]

    ep = neo.Epoch(times=times, durations=durations, **kwargs)

    ep.annotate(**event1.annotations)

    if attach_result:
        segment.epochs.append(ep)
        segment.create_relationship()

    return ep


def match_events(event1, event2):
    """
    Finds pairs of Event entries in event1 and event2 with the minimum delay,
    such that the entry of event1 directly precedes the entry of event2.
    Returns filtered two events of identical length, which contain matched
    entries.

    Parameters:
    -----------
    event1, event2: neo.Event
        The two Event objects to match up.

    Returns:
    --------
    event1, event2: neo.Event
        Event objects with identical number of events, containing only those
        events that could be matched against each other. A warning is issued if
        not all events in event1 or event2 could be matched.
    """
    event1 = event1
    event2 = event2

    id1, id2 = 0, 0
    match_ev1, match_ev2 = [], []
    while id1 < len(event1) and id2 < len(event2):
        time1 = event1.times[id1]
        time2 = event2.times[id2]

        # wrong order of events
        if time1 >= time2:
            id2 += 1

        # shorter epoch possible by later event1 entry
        elif id1 + 1 < len(event1) and event1.times[id1 + 1] < time2:
                # there is no event in 2 until the next event in 1
            id1 += 1

        # found a match
        else:
            match_ev1.append(id1)
            match_ev2.append(id2)
            id1 += 1
            id2 += 1

    if id1 < len(event1):
        warnings.warn(
            'Could not match all events to generate epochs. Missed '
            '%s event entries in event1 list' % (len(event1) - id1))
    if id2 < len(event2):
        warnings.warn(
            'Could not match all events to generate epochs. Missed '
            '%s event entries in event2 list' % (len(event2) - id2))

    event1_matched = _event_epoch_slice_by_valid_ids(
        obj=event1, valid_ids=match_ev1)
    event2_matched = _event_epoch_slice_by_valid_ids(
        obj=event2, valid_ids=match_ev2)

    return event1_matched, event2_matched


def cut_block_by_epochs(block, properties=None, reset_time=False):
    """
    This function cuts Neo Segments in a Neo Block according to multiple Neo
    Epoch objects.

    The function alters the Neo Block by adding one Neo Segment per Epoch entry
    fulfilling a set of conditions on the Epoch attributes and annotations. The
    original segments are removed from the block.

    A dictionary contains restrictions on which epochs are considered for
    the cutting procedure. To this end, it is possible to
    specify accepted (valid) values of specific annotations on the source
    epochs.

    The resulting cut segments may either retain their original time stamps, or
    be shifted to a common starting time.

    Parameters
    ----------
    block: Neo Block
        Contains the Segments to cut according to the Epoch criteria provided
    properties: dictionary
        A dictionary that contains the Epoch keys and values to filter for.
        Each key of the dictionary is matched to an attribute or an an
        annotation of the Event. The value of each dictionary entry corresponds
        to a valid entry or a list of valid entries of the attribute or
        annotation.

        If the value belonging to the key is a list of entries of the same
        length as the number of epochs in the Epoch object, the list entries
        are matched to the epochs in the Epoch object. The resulting Epoch
        object contains only those epochs where the values match up.

        Otherwise, the value is compared to the attributes or annotation of the
        Epoch object as such, and depending on the comparison, either the
        complete Epoch object is returned or not.

        If None or an empty dictionary is passed, all Epoch Objects will
        be considered

    reset_time: bool
        If True the times stamps of all sliced objects are set to fall
        in the range from 0 to the duration of the epoch duration.
        If False, original time stamps are retained.
        Default is False.

    Returns:
    --------
    None
    """
    if not isinstance(block, neo.Block):
        raise TypeError(
            'block needs to be a neo Block, not %s' % type(block))

    old_segments = copy.copy(block.segments)
    for seg in old_segments:
        epochs = _get_from_list(seg.epochs, prop=properties)
        if len(epochs) > 1:
            warnings.warn(
                'Segment %s contains multiple epochs with '
                'requested properties (%s). Subsegments can '
                'have overlapping times' % (seg.name, properties))

        elif len(epochs) == 0:
            warnings.warn(
                'No epoch is matching the requested epoch properties %s. '
                'No cutting of segment %s performed.' % (properties, seg.name))

        for epoch in epochs:
            new_segments = cut_segment_by_epoch(
                seg, epoch=epoch, reset_time=reset_time)
            block.segments += new_segments

        block.segments.remove(seg)
    block.create_many_to_one_relationship(force=True)


def cut_segment_by_epoch(seg, epoch, reset_time=False):
    """
    Cuts a Neo Segment according to a neo Epoch object

    The function returns a list of neo Segments, where each segment corresponds
    to an epoch in the neo Epoch object and contains the data of the original
    Segment cut to that particular Epoch.

    The resulting segments may either retain their original time stamps,
    or can be shifted to a common time axis.

    Parameters
    ----------
    seg: Neo Segment
        The Segment containing the original uncut data.
    epoch: Neo Epoch
        For each epoch in this input, one segment is generated according to
         the epoch time and duration.
    reset_time: bool
        If True the times stamps of all sliced objects are set to fall
        in the range from 0 to the duration of the epoch duration.
        If False, original time stamps are retained.
        Default is False.

    Returns:
    --------
    segments: list of Neo Segments
        Per epoch in the input, a neo.Segment with AnalogSignal and/or
        SpikeTrain Objects will be generated and returned. Each Segment will
        receive the annotations of the corresponding epoch in the input.
    """
    if not isinstance(seg, neo.Segment):
        raise TypeError(
            'Seg needs to be of type neo.Segment, not %s' % type(seg))

    if type(seg.parents[0]) != neo.Block:
        raise ValueError(
            'Segment has no block as parent. Can not cut segment.')

    if not isinstance(epoch, neo.Epoch):
        raise TypeError(
            'Epoch needs to be of type neo.Epoch, not %s' % type(epoch))

    segments = []
    for ep_id in range(len(epoch)):
        subseg = seg_time_slice(seg,
                                epoch.times[ep_id],
                                epoch.times[ep_id] + epoch.durations[ep_id],
                                reset_time=reset_time)

        # Add annotations of Epoch
        for a in epoch.annotations:
            if type(epoch.annotations[a]) is list \
                    and len(epoch.annotations[a]) == len(epoch):
                subseg.annotations[a] = copy.copy(epoch.annotations[a][ep_id])
            else:
                subseg.annotations[a] = copy.copy(epoch.annotations[a])

        # Add array-annotations of Epoch
        for key, val in epoch.array_annotations.items():
            if len(val):
                subseg.annotations[key] = copy.copy(val[ep_id])

        segments.append(subseg)

    return segments


def seg_time_slice(seg, t_start=None, t_stop=None, reset_time=False, **kwargs):
    """
    Creates a time slice of a neo Segment containing slices of all child
    objects.

    Parameters:
    -----------
    seg: neo Segment
        The neo Segment object to slice.
    t_start: Quantity
        Starting time of the sliced time window.
    t_stop: Quantity
        Stop time of the sliced time window.
    reset_time: bool
        If True the times stamps of all sliced objects are set to fall
        in the range from 0 to the duration of the epoch.
        If False, original time stamps are retained.
        Default is False.

    Keyword Arguments:
    ------------------
        Additional keyword arguments used for initialization of the sliced
        Neo Segment object.

    Returns:
    --------
    seg: Neo Segment
        Temporal slice of the original Neo Segment from t_start to t_stop.
    """
    subseg = neo.Segment(**kwargs)

    for attr in [
            'file_datetime', 'rec_datetime', 'index',
            'name', 'description', 'file_origin']:
        setattr(subseg, attr, getattr(seg, attr))

    subseg.annotations = copy.deepcopy(seg.annotations)

    t_shift = - t_start

    # cut analogsignals and analogsignalarrays
    for ana_id in range(len(seg.analogsignals)):
        ana_time_slice = seg.analogsignals[ana_id].time_slice(t_start, t_stop)
        if reset_time:
            ana_time_slice.t_start = ana_time_slice.t_start + t_shift
        subseg.analogsignals.append(ana_time_slice)

    # cut spiketrains
    for st_id in range(len(seg.spiketrains)):
        st_time_slice = seg.spiketrains[st_id].time_slice(t_start, t_stop)
        if reset_time:
            st_time_slice = shift_spiketrain(st_time_slice, t_shift)
        subseg.spiketrains.append(st_time_slice)

    # cut events
    for ev_id in range(len(seg.events)):
        ev_time_slice = event_time_slice(seg.events[ev_id], t_start, t_stop)
        if reset_time:
            ev_time_slice = shift_event(ev_time_slice, t_shift)
        # appending only non-empty events
        if len(ev_time_slice):
            subseg.events.append(ev_time_slice)

    # cut epochs
    for ep_id in range(len(seg.epochs)):
        ep_time_slice = epoch_time_slice(seg.epochs[ep_id], t_start, t_stop)
        if reset_time:
            ep_time_slice = shift_epoch(ep_time_slice, t_shift)
        # appending only non-empty epochs
        if len(ep_time_slice):
            subseg.epochs.append(ep_time_slice)

    return subseg


def shift_spiketrain(spiketrain, t_shift):
    """
    Shifts a spike train to start at a new time.

    Parameters:
    -----------
    spiketrain: Neo SpikeTrain
        Spiketrain of which a copy will be generated with shifted spikes and
        starting and stopping times
    t_shift: Quantity (time)
        Amount of time by which to shift the SpikeTrain.

    Returns:
    --------
    spiketrain: Neo SpikeTrain
        New instance of a SpikeTrain object starting at t_start (the original
        SpikeTrain is not modified).
    """
    new_st = spiketrain.duplicate_with_new_data(
        signal=spiketrain.times.view(pq.Quantity) + t_shift,
        t_start=spiketrain.t_start + t_shift,
        t_stop=spiketrain.t_stop + t_shift)
    return new_st


def event_time_slice(event, t_start=None, t_stop=None):
    """
    Slices an Event object to retain only those events that fall in a certain
    time window.

    Parameters:
    -----------
    event: Neo Event
        The Event to slice.
    t_start, t_stop: Quantity (time)
        Time window in which to retain events. An event at time t is retained
        if t_start <= t < t_stop.

    Returns:
    --------
    event: Neo Event
        New instance of an Event object containing only the events in the time
        range.
    """
    if t_start is None:
        t_start = -np.inf
    if t_stop is None:
        t_stop = np.inf

    valid_ids = np.where(np.logical_and(
        event.times >= t_start, event.times < t_stop))[0]

    new_event = _event_epoch_slice_by_valid_ids(event, valid_ids=valid_ids)

    return new_event


def epoch_time_slice(epoch, t_start=None, t_stop=None):
    """
    Slices an Epoch object to retain only those epochs that fall in a certain
    time window.

    Parameters:
    -----------
    epoch: Neo Epoch
        The Epoch to slice.
    t_start, t_stop: Quantity (time)
        Time window in which to retain epochs. An epoch at time t and
        duration d is retained if t_start <= t < t_stop - d.

    Returns:
    --------
    epoch: Neo Epoch
        New instance of an Epoch object containing only the epochs in the time
        range.
    """
    if t_start is None:
        t_start = -np.inf
    if t_stop is None:
        t_stop = np.inf

    valid_ids = np.where(np.logical_and(
        epoch.times >= t_start, epoch.times + epoch.durations < t_stop))[0]

    new_epoch = _event_epoch_slice_by_valid_ids(epoch, valid_ids=valid_ids)

    return new_epoch


def shift_event(ev, t_shift):
    """
    Shifts an event by an amount of time.

    Parameters:
    -----------
    event: Neo Event
        Event of which a copy will be generated with shifted times
    t_shift: Quantity (time)
        Amount of time by which to shift the Event.

    Returns:
    --------
    epoch: Neo Event
        New instance of an Event object starting at t_shift later than the
        original Event (the original Event is not modified).
    """
    return _shift_time_signal(ev, t_shift)


def shift_epoch(epoch, t_shift):
    """
    Shifts an epoch by an amount of time.

    Parameters:
    -----------
    epoch: Neo Epoch
        Epoch of which a copy will be generated with shifted times
    t_shift: Quantity (time)
        Amount of time by which to shift the Epoch.

    Returns:
    --------
    epoch: Neo Epoch
        New instance of an Epoch object starting at t_shift later than the
        original Epoch (the original Epoch is not modified).
    """
    return _shift_time_signal(epoch, t_shift)


def _shift_time_signal(sig, t_shift):
    """
    Internal function.
    """
    if not hasattr(sig, 'times'):
        raise AttributeError(
            'Can only shift signals, which have an attribute'
            ' "times", not %s' % type(sig))
    new_sig = sig.duplicate_with_new_data(signal=sig.times + t_shift)
    return new_sig
