from gym.envs.registration import register

register(
    id='sawyer-peg-v0',
    entry_point='gym_tacto.envs:SawyerPegV0',
)

register(
    id='sawyer-peg-v1',
    entry_point='gym_tacto.envs:SawyerPegV1',
)