from neo import Group, Unit, ChannelView, SpikeTrain
from neo.core.basesignal import BaseSignal


def _convert_unit(unit):
    group_unit = Group(unit.spiketrains,
                       name=unit.name,
                       file_origin=unit.file_origin,
                       description=unit.description,
                       allowed_types=[SpikeTrain],
                       **unit.annotations)
    # clean up references
    for st in unit.spiketrains:
        delattr(st, 'unit')
    return group_unit


def _convert_channel_index(channel_index):
    # convert child objects
    new_child_objects = []
    for child_obj in channel_index.children:
        if isinstance(child_obj, Unit):
            new_unit = _convert_unit(child_obj)
            new_child_objects.append(new_unit)
        elif isinstance(child_obj, BaseSignal):
            # always generate view, as this might provide specific info regarding the object
            new_view = ChannelView(child_obj, channel_index.index,
                                   name=channel_index.name,
                                   description=channel_index.description,
                                   file_origin=channel_index.file_origin,
                                   **channel_index.annotations)
            new_view.array_annotate(channel_ids=channel_index.channel_ids,
                                    channel_names=channel_index.channel_names)

            # separate dimenions of coordinates into different 1D array_annotations
            if channel_index.coordinates.shape:
                if len(channel_index.coordinates.shape) == 1:
                    new_view.array_annotate(coordinates=channel_index.coordinates)
                elif len(channel_index.coordinates.shape) == 2:
                    for dim in range(channel_index.coordinates.shape[1]):
                        new_view.array_annotate(
                            **{f'coordinates_dim{dim}': channel_index.coordinates[:, dim]})
                else:
                    raise ValueError(f'Incompatible channel index coordinates with wrong '
                                     f'dimensions: Provided coordinates have shape '
                                     f'{channel_index.coordinates.shape}.')

            # clean up references
            delattr(child_obj, 'channel_index')

            new_child_objects.append(new_view)

    new_channel_group = Group(new_child_objects,
                              name=channel_index.name,
                              file_origin=channel_index.file_origin,
                              description=channel_index.description,
                              **channel_index.annotations)

    return new_channel_group


def convert_channelindex_to_view_group(block):
    """
    Convert deprecated ChannelIndex and Unit objects to ChannelView and Group objects

    This conversion is preserving all information stored as attributes and (array) annotations.
    The conversion is done in-place.
    Each ChannelIndex is represented as a Group. Linked Unit objects are represented as child Group
    (subgroup) objects. Linked data objects (neo.AnalogSignal, neo.IrregularlySampledSignal) are
    represented by a View object linking to the original data object.
    Attributes are as far as possible conserved by the conversion of objects. `channel_ids`,
    `channel_names` and `coordinates` are converted to array_annotations.

    :param block: neo.Block structure to be converted
    :return: block: updated neo.Block structure
    """
    for channel_index in block.channel_indexes:
        new_channel_group = _convert_channel_index(channel_index)
        block.groups.append(new_channel_group)

    # clean up references
    delattr(block, 'channel_indexes')

    # this is a hack to clean up ImageSequence objects that are not properly linked to
    # ChannelIndex objects, see also Issue #878
    for seg in block.segments:
        for imgseq in seg.imagesequences:
            delattr(imgseq, 'channel_index')

    return block
