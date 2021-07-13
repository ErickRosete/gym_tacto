import gym_tacto
from pathlib                 import Path
from hydra.utils             import get_original_cwd
from hydra.core.hydra_config import HydraConfig

def get_file_list(data_dir):
    """ retrieve a list of files inside a folder """
    dir_path = Path(data_dir)
    assert dir_path.is_dir()
    file_list = []
    for x in dir_path.iterdir():
        if x.is_file():
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(get_file_list(x))
    return file_list

def get_cwd():
    if HydraConfig.initialized():
        cwd = Path(get_original_cwd())
    else:
        cwd = Path.cwd()   
    return cwd

def add_cwd(path):
    return str((get_cwd() / path).resolve())

def pkg_path(rel_path):
    """Generates a global path that is relative to
    the root of drlfads package.
    (Could be generalized for any python module)

    Args:
        rel_path (str): Relative path within drlfads package 

    Returns:
        str: Global path.
    """
    return str(Path(gym_tacto.__path__[0], rel_path))