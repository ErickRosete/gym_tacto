from gym.envs.registration import register

register(
    id='sawyer-peg-v0',
    entry_point='gym_tacto.envs:SawyerPegV0',
    max_episode_steps=500
)

register(
    id='sawyer-peg-v1',
    entry_point='gym_tacto.envs:SawyerPegV1',
    max_episode_steps=500
)

register(
    id='sawyer-door-v0',
    entry_point='gym_tacto.envs:SawyerDoorV0',
    max_episode_steps=500
)