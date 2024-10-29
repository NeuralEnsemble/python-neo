"""
Experimental module to export a rawio reader that support the buffer_description API
to xarray dataset using zarr specification format v2.

A block/segment/stream correspond to one xarray.DataSet

A xarray.DataTree can also be expose to get all at block/segment/stream

Note :
  * only some IOs support this at the moment has_buffer_description_api()=True
"""

import numpy as np

import base64


def to_zarr_v2_reference(rawio_reader, block_index=0, seg_index=0, buffer_id=None):
    """
    Transform the buffer_description_api into a dict ready for the xarray API 'reference://'


    See https://fsspec.github.io/kerchunk/spec.html
    See https://docs.xarray.dev/en/latest/user-guide/io.html#kerchunk

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html


    Usefull read also https://github.com/saalfeldlab/n5

    """

    # TODO later implement zarr v3

    # rawio_reader.
    signal_buffers = rawio_reader.header["signal_buffers"]

    buffer_index = list(signal_buffers["id"]).index(buffer_id)

    buffer_name = signal_buffers["name"][buffer_index]

    rfs = dict()
    rfs["version"] = 1
    rfs["refs"] = dict()
    rfs["refs"][".zgroup"] = dict(zarr_format=2)
    zattrs = dict(name=buffer_name)
    rfs["refs"][".zattrs"] = zattrs

    descr = rawio_reader.get_analogsignal_buffer_description(
        block_index=block_index, seg_index=seg_index, buffer_id=buffer_id
    )

    if descr["type"] == "raw":

        # channel : small enough can be internal with base64
        mask = rawio_reader.header["signal_channels"]["buffer_id"] == buffer_id
        channels = rawio_reader.header["signal_channels"][mask]
        channel_ids = channels["id"]
        base64_encoded = base64.b64encode(channel_ids.tobytes())
        rfs["refs"]["channel/0"] = "base64:" + base64_encoded.decode()
        zarray = dict(
            chunks=channel_ids.shape,
            compressor=None,
            dtype=channel_ids.dtype.str,
            fill_value=None,
            filters=None,
            order="C",
            shape=channel_ids.shape,
            zarr_format=2,
        )
        zattrs = dict(
            _ARRAY_DIMENSIONS=["channel"],
        )
        rfs["refs"]["channel/.zattrs"] = zattrs
        rfs["refs"]["channel/.zarray"] = zarray

        # traces buffer
        dtype = np.dtype(descr["dtype"])
        zarray = dict(
            chunks=descr["shape"],
            compressor=None,
            dtype=dtype.str,
            fill_value=None,
            filters=None,
            order=descr["order"],
            shape=descr["shape"],
            zarr_format=2,
        )
        zattrs = dict(
            _ARRAY_DIMENSIONS=["time", "channel"],
            name=buffer_name,
        )
        units = np.unique(channels["units"])
        if units.size == 1:
            zattrs["units"] = units[0]
        gain = np.unique(channels["gain"])
        offset = np.unique(channels["offset"])
        if gain.size == 1 and offset.size:
            zattrs["scale_factor"] = gain[0]
            zattrs["add_offset"] = offset[0]
        zattrs["sampling_rate"] = float(channels["sampling_rate"][0])

        # unique big chunk
        # TODO later : optional split in several small chunks
        array_size = np.prod(descr["shape"], dtype="int64")
        chunk_size = array_size * dtype.itemsize
        rfs["refs"]["traces/0.0"] = [str(descr["file_path"]), descr["file_offset"], chunk_size]
        rfs["refs"]["traces/.zarray"] = zarray
        rfs["refs"]["traces/.zattrs"] = zattrs

    elif descr["type"] == "hdf5":
        raise NotImplementedError
    else:
        raise ValueError(f"buffer_description type not handled {descr['type']}")

    # TODO later channel array_annotations

    return rfs


def to_xarray_dataset(rawio_reader, block_index=0, seg_index=0, buffer_id=None):
    """
    Utils fonction that transorm an instance a rawio into a xarray.Dataset
    with lazy access.
    This works only for rawio class that return True with has_buffer_description_api() and hinerits from
    BaseRawWithBufferApiIO.


    Note : the original idea of the function is from Ben Dichter in this page
    https://gist.github.com/bendichter/30a9afb34b2178098c99f3b01fe72e75
    """
    import xarray as xr

    rfs = to_zarr_v2_reference(rawio_reader, block_index=block_index, seg_index=seg_index, buffer_id=buffer_id)

    ds = xr.open_dataset(
        "reference://",
        mask_and_scale=True,
        engine="zarr",
        backend_kwargs={
            "storage_options": dict(
                fo=rfs,
                remote_protocol="file",
            ),
            "consolidated": False,
        },
    )
    return ds


def to_xarray_datatree(rawio_reader):
    """
    Expose a neo.rawio reader class to a xarray DataTree to lazily read signals.
    """
    try:
        # not released in xarray 2024.7.0, this will be released soon
        from xarray import DataTree
    except:
        # this need the experimental DataTree in pypi xarray-datatree
        try:
            from datatree import DataTree
        except:
            raise ImportError("use xarray dev branch or pip install xarray-datatree")

    signal_buffers = rawio_reader.header["signal_buffers"]
    buffer_ids = signal_buffers["id"]

    tree = DataTree(name="root")

    num_block = rawio_reader.header["nb_block"]
    for block_index in range(num_block):
        block = DataTree(name=f"block{block_index}", parent=tree)
        num_seg = rawio_reader.header["nb_segment"][block_index]
        for seg_index in range(num_seg):
            segment = DataTree(name=f"segment{seg_index}", parent=block)
            for buffer_id in buffer_ids:
                ds = to_xarray_dataset(rawio_reader, block_index=block_index, seg_index=seg_index, buffer_id=buffer_id)
                DataTree(data=ds, name=ds.attrs["name"], parent=segment)

    return tree
