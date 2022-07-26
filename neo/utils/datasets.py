"""
Utility functions to retrieve public datasets.
"""
import os
from pathlib import Path

default_testing_repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

global local_testing_data_folder
if os.getenv('EPHY_TESTING_DATA_FOLDER', default=None) is not None:
    local_testing_data_folder = Path(os.getenv('EPHY_TESTING_DATA_FOLDER'))
else:
    # set in home
    local_testing_data_folder = Path.home() / 'ephy_testing_data'


def get_local_testing_data_folder():
    global local_testing_data_folder
    return local_testing_data_folder


def download_dataset(repo=default_testing_repo, remote_path=None,
                     local_folder=None):
    """
    Download a dataset with datalad client.

    By default it download the "NeuralEnsemble/ephy_testing_data" on gin platform
    which is used for neo testing.

    Usage:

        download_dataset(
                repo='https://gin.g-node.org/NeuralEnsemble/ephy_testing_data',
                remote_path='blackrock/blackrock_2_1',
                local_folder='/home/myname/Documents/')

    Parameters
    ----------
    repo: str
        The url of the repo.
        If None then 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
        is used
    remote_path: str of Path
        The distant path to retrieve (file or folder)
    local_folder: str or Path or None
        The local folder where to download the data.
        If None, a default project testing folder is used. Default: None

    Returns
    -------
    local_path:
        The local path of the downloaded file or folder
    """
    import datalad.api
    from datalad.support.gitrepo import GitRepo

    if local_folder is None:
        global local_testing_data_folder
        local_folder = local_testing_data_folder
    local_folder = Path(local_folder)

    if local_folder.exists() and GitRepo.is_valid_repo(local_folder):
        dataset = datalad.api.Dataset(path=local_folder)
        # make sure git repo is in clean state
        repo = dataset.repo
        repo.call_git(['checkout', '--force', 'master'])
        dataset.update(merge=True)
    else:
        dataset = datalad.api.install(path=local_folder,
                                      source=repo)

    if remote_path is None:
        print('Bad boy: you have to provide "remote_path"')
        return

    dataset.get(remote_path)

    local_path = local_folder / remote_path

    return local_path
