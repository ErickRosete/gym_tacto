# Gym Tacto 
Set of PyBullet environments for robotic manipulation that can use tactile information

## Installation
The package was tested using python 3.7 and Ubuntu 20.04 LTS. <br/>
To install use the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
$ git clone https://github.com/ErickRosete/gym_tacto.git
$ cd gym_tacto
$ pip install -e .
```
## Sawyer Grasp Environmet
TODO description and photo
[hydra env configuration file](./config/env/sawyer_grasp_env.yaml). 

## Sawyer Door Environmet
TODO description and photo
[hydra env configuration file](./config/env/sawyer_door_env.yaml). 

## Sawyer Peg Environment
TODO add photo
The goal of the task is to insert a peg into a box, the environment works similar to Gym from OpenAI environments.
```python

@hydra.main(config_path="../config", config_name="sac_gmm_config")
def main(cfg):
    env = gym.make('gym_tacto:sawyer-peg-v1')
    for episode in range(100):
        observation = env.reset()
        episode_length, episode_reward = 0,0
        for step in range(100):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

if __name__ == "__main__":
    main()
```
The environment observation consists in a dictionary with different values that could be accessed when requested in the [hydra env configuration file](./config/env/sawyer_peg_env.yaml). <br/>
All the possible parameters are listed below
```python
        observation_space["position"]
        observation_space["gripper_width"]
        observation_space["tactile_sensor"]
        observation_space["force"]
```
- "position" contains the end effector cartesian pose
- "gripper_width" defines the end effector distance between each finger
- "tactile_sensor" contains the image measurement of each finger
- "force" contains the force readings in each finger 
