import unittest

from neo.utils import download_dataset

class TestDownloadDataset(unittest.TestCase):
    def test_download_dataset(self):
        local_path = download_dataset(
                repo='https://gin.g-node.org/NeuralEnsemble/ephy_testing_data',
                remote_path='blackrock/blackrock_2_1')
        assert local_path.is_dir()


if __name__ == "__main__":
    unittest.main()
