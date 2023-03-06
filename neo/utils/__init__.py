'''
Utils package
'''

from .misc import (get_events, get_epochs, add_epoch,
    match_events, cut_block_by_epochs, cut_segment_by_epoch,
    is_block_rawio_compatible)
from .datasets import download_dataset, get_local_testing_data_folder
