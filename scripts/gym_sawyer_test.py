import hydra
from gym_tacto.utils.gym_utils import make_env

@hydra.main(config_path="../config", config_name="env_test")
def env_test(cfg):
    env = make_env(cfg.env.name, cfg.env)
    for episode in range(100):
        s = env.reset()
        episode_length, episode_reward = 0,0
        for step in range(100):
            a = env.action_space.sample()
            s, r, done, _ = env.step(a)
            if step % 100 == 0:
                print("Action", a)
                if isinstance(s, dict):
                    print("Position", s["position"])
                else:
                    print("State", s)
            if done:
                break

if __name__ == "__main__":
    env_test()