import json

import numpy as np



def to_xarray_reference_api(rawio_reader, block_index=0, seg_index=0):
    """
    Transform the buffer_description_api into a dict ready for the xarray API 'reference://'
    """
    rfs = dict()
    rfs["version"] = 1
    rfs["refs"] = dict()
    rfs["refs"][".zgroup"] = json.dumps(dict(zarr_format=2))

    # rawio_reader.
    signal_streams = rawio_reader.header["signal_streams"]
    for stream_index in range(len(signal_streams)):
        stream_name = signal_streams["name"][stream_index]
        stream_id = signal_streams["id"][stream_index]
        print(stream_index, stream_name, stream_id)
        
        
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
                _ARRAY_DIMENSIONS=[f'time_{stream_id}', f'channel_{stream_id}'],
            )
            # unique big chunk
            # TODO : optional split in several small chunks
            array_size = np.prod(descr["shape"], dtype='int64')
            chunk_size = array_size * dtype.itemsize
            rfs["refs"][f"{stream_id}/0.0"] = [str(descr["file_path"]), descr["file_offset"], chunk_size]
            rfs["refs"][f"{stream_id}/.zarray"] =json.dumps(zarray)
            rfs["refs"][f"{stream_id}/.zattrs"] =json.dumps(zattrs)
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



def to_xarray_dataset(rawio_reader):
    """
    Utils fonction that transorm an instance a rawio into a xarray.Dataset
    with lazy access.
    This works only for rawio class that implement the has_buffer_description_api.
    

    Note : the original idea of the function is from Ben Dichter in this page
    https://gist.github.com/bendichter/30a9afb34b2178098c99f3b01fe72e75
    """
    import xarray as xr

    rfs = to_xarray_reference_api(rawio_reader)

    ds = xr.open_dataset(
        "reference://",
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


