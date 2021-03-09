from gym.envs.registration import register

register(
    id='codeside-v0',
    entry_point='codeside.envs:CodeSideEnv',
)
