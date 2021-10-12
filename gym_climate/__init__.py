from gym.envs.registration import register

register(
    id='dice-v0',
    entry_point='gym_climate.envs.DICE:EnvDICE'
)

register(
    id="ays-v0",
    entry_point="gym_climate.envs.AYS:AYSEnvironment"
)
