import importlib

def make_env(env_id, cfg):
    get_env = getattr(importlib.import_module("gym_tacto.envs.%s" % env_id), "make_env_%s" % env_id)
    env = get_env(cfg)
    return env

def make_env_fn(env_id, cfg, seed):
    def _f():
        env = make_env(env_id, cfg)
        env.seed(seed)
        return env
    return _f
