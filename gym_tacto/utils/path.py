import gym_tacto
from pathlib import Path


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