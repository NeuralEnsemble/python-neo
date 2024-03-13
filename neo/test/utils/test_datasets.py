import unittest
import pytest
from neo.utils.datasets import download_dataset, default_testing_repo
from neo.test.rawiotest.tools import can_use_network


@pytest.mark.skipif(not can_use_network(), reason="Must have acess to network to run test")
class TestDownloadDataset(unittest.TestCase):
    def test_download_dataset(self):
        local_path = download_dataset(repo=default_testing_repo, remote_path="blackrock/blackrock_2_1")
        assert local_path.is_dir()


if __name__ == "__main__":
    unittest.main()
