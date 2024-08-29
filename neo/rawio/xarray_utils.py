import json

import numpy as np

import base64



def to_xarray_reference_api(rawio_reader, block_index=0, seg_index=0, stream_index=0):
    """
    Transform the buffer_description_api into a dict ready for the xarray API 'reference://'


    See https://fsspec.github.io/kerchunk/spec.html
    See https://docs.xarray.dev/en/latest/user-guide/io.html#kerchunk


    """
    rfs = dict()
    rfs["version"] = 1
    rfs["refs"] = dict()
    rfs["refs"][".zgroup"] = json.dumps(dict(zarr_format=2))

    # rawio_reader.
    signal_streams = rawio_reader.header["signal_streams"]

    stream_name = signal_streams["name"][stream_index]
    stream_id = signal_streams["id"][stream_index]
    # print(stream_index, stream_name, stream_id)
    
    
    descr = rawio_reader.get_analogsignal_buffer_description(block_index=block_index, seg_index=seg_index, 
                                                                stream_index=stream_index)

    if descr["type"] == "binary":
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
            _ARRAY_DIMENSIONS=['time', 'channel'],
            name=stream_name,
            stream_id=stream_id,
        )
        # unique big chunk
        # TODO : optional split in several small chunks
        array_size = np.prod(descr["shape"], dtype='int64')
        chunk_size = array_size * dtype.itemsize
        rfs["refs"]["traces/0.0"] = [str(descr["file_path"]), descr["file_offset"], chunk_size]
        rfs["refs"]["traces/.zarray"] =json.dumps(zarray)
        rfs["refs"]["traces/.zattrs"] =json.dumps(zattrs)

        # small enough can be internal
        mask = rawio_reader.header["signal_channels"]["stream_id"] == stream_id
        channel_ids = rawio_reader.header["signal_channels"][mask]["id"]
        base64_encoded = base64.b64encode(channel_ids.tobytes())
        rfs["refs"]["yep/0"] = "base64:" + base64_encoded.decode()
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
            _ARRAY_DIMENSIONS=['channel'],
        )
        rfs["refs"]["yep/.zattrs"] =json.dumps(zattrs)
        rfs["refs"]["yep/.zarray"] =json.dumps(zarray)

    elif descr["type"] == "hdf5":
        raise NotImplementedError
    else:
        raise ValueError(f"buffer_description type not handled {descr['type']}")



    # TODO magnitude gain and offset
    # TODO channel names
    # TODO sampling rate
    # TODO annotations
    # TODO array_annotations


    return rfs



def to_xarray_dataset(rawio_reader, block_index=0, seg_index=0, stream_index=0):
    """
    Utils fonction that transorm an instance a rawio into a xarray.Dataset
    with lazy access.
    This works only for rawio class that implement the has_buffer_description_api.
    

    Note : the original idea of the function is from Ben Dichter in this page
    https://gist.github.com/bendichter/30a9afb34b2178098c99f3b01fe72e75
    """
    import xarray as xr

    rfs = to_xarray_reference_api(rawio_reader, block_index=block_index, seg_index=seg_index, stream_index=stream_index)

    ds = xr.open_dataset(
        "reference://",
        # mask_and_scale=True,
        mask_and_scale=False,
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


