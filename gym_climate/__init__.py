from gym.envs.registration import register

register(
    id='dice-v0',
    entry_point='gym_climate.envs.DICE:EnvDICE'
)

register(
    id='AYS-v0',
    entry_point='gym_climate.envs.AYS:AYS_Environment'
)

register(
    id='AYS-v1',
    entry_point='gym_climate.envs.AYS:noisy_partially_observable_AYS'
)

register(
    id='c_global-v0',
    entry_point='gym_climate.envs.c_global:cG_LAGTPKS_Environment'
)

register(
    id='c_global-v1',
    entry_point='gym_climate.envs.c_global:partially_observable_cG'
)

register(
    id='c_global-v2',
    entry_point='gym_climate.envs.c_global:noisy_partially_observable_cG'
)
