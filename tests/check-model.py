import random

import gym
from stable_baselines3.common.env_checker import check_env
from test_DICE2016 import *

import gym_climate

for i in range(10):
    action_scalar = random.random()

    # RUNNING PYDICE MODEL TO OUTPUT A CSV OF PARAMETERS FOR COMPARISON
    TT = np.linspace(2000, 2500, 100, dtype=np.int32)

    InitializeLabor(l, NT)
    InitializeTFP(al, NT)
    InitializeGrowthSigma(gsig, NT)
    InitializeSigma(sigma, gsig, cost1, NT)
    InitializeCarbonTree(cumetree, NT)

    x_start = np.concatenate([MIU_start, S_start])
    bnds = bnds1 + bnds2
    x = np.ones(2 * NT) * action_scalar
    test_result = fOBJ(
        x,
        1.0,
        I,
        K,
        al,
        l,
        YGROSS,
        sigma,
        EIND,
        E,
        CCA,
        CCATOT,
        cumetree,
        MAT,
        MU,
        ML,
        FORC,
        TATM,
        TOCEAN,
        DAMFRAC,
        DAMAGES,
        ABATECOST,
        cost1,
        MCABATE,
        CPRICE,
        YNET,
        Y,
        C,
        CPC,
        PERIODU,
        CEMUTOTPER,
        RI,
        NT,
    )

    # RUNNING DICE ENV
    env = gym.make("dice-v0")
    check_env(env)
    env.test_flag = True
    obs = env.reset()
    rewards = 0
    renormed_action = action_scalar * 2 - 1
    for i in range(99):
        state, reward, done, _ = env.step(
            np.array([renormed_action, renormed_action])
        )
    assert str(test_result)[:8] == str(reward)[:8], "Model error"
