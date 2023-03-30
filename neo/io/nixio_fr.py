from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.nixrawio import NIXRawIO
import warnings

# This class subjects to limitations when there are multiple asymmetric blocks


class NixIO(NIXRawIO, BaseFromRaw):

    name = 'NIX IO'

    _prefered_signal_group_mode = 'group-by-same-units'
    _prefered_units_group_mode = 'all-in-one'

    def __init__(self, filename, block_index=0, autogenerate_stream_names=False, autogenerate_unit_ids=False):
        NIXRawIO.__init__(self, filename,
                          block_index=block_index,
                          autogenerate_stream_names=autogenerate_stream_names,
                          autogenerate_unit_ids=autogenerate_unit_ids)
        BaseFromRaw.__init__(self, filename)

    def read_block(self, block_index=0, **kwargs):
        # sanity check to ensure constructed header and block to load match
        if block_index != 0:
            warnings.warn(f'Initialized IO for block {self.block_index}.'
                          f'Can only read that block. Ignoring additional {block_index=} argument.')

        return super(NixIO, self).read_block(block_index=0, **kwargs)

    def read_segment(self, block_index=0, **kwargs):
        # sanity check to ensure constructed header and block to load match
        if block_index != 0:
            warnings.warn(f'Initialized IO for block {self.block_index}.'
                          f'Can only read that block. Ignoring additional {block_index=} argument.')

        return super(NixIO, self).read_segment(block_index=0, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.header = None
        self.file.close()
