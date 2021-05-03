import unittest

from neo.utils.datasets import download_dataset, default_testing_repo


class TestDownloadDataset(unittest.TestCase):
    def test_download_dataset(self):
        local_path = download_dataset(
            repo=default_testing_repo,
            remote_path='blackrock/blackrock_2_1')
        assert local_path.is_dir()


if __name__ == "__main__":
    unittest.main()
