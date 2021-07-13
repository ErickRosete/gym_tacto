import importlib
from gym_tacto.utils.gym_utils import make_env

def get_pd_controller(cfg):
    env = make_env(cfg.env.name, cfg.env)
    PDController = getattr(importlib.import_module("gym_tacto.PD.pd_%s" % cfg.env.name), "PDController")
    pd = PDController(env)
    return pd