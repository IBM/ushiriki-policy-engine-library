from gym.envs.registration import register

register(
    id='ChallengeAction-v0',
    entry_point='ushiriki_policy_engine_library.envs:ChallengeActionEnv',
    max_episode_steps=2000,
)

register(
    id='ChallengePolicy-v0',
    entry_point='ushiriki_policy_engine_library.envs:ChallengePolicyEnv',
    max_episode_steps=400,
)

register(
    id='ICLR23Workshop-env1-v0',
    entry_point='ushiriki_policy_engine_library.ICLR23ChallengeEnvironment:Model0Env_beta',
)

register(
    id='ICLR23Workshop-env2-v0',
    entry_point='ushiriki_policy_engine_library.ICLR23ChallengeEnvironment:Model0Env_batchbeta',
)

register(
    id='ICLR23Workshop-env3-v0',
    entry_point='ushiriki_policy_engine_library.ICLR23ChallengeEnvironment:Model0Env_betalist',
)

register(
    id='ICLR23Workshop-env4-v0',
    entry_point='ushiriki_policy_engine_library.ICLR23ChallengeEnvironment:Model1Env_beta',
)

register(
    id='ICLR23Workshop-env5-v0',
    entry_point='ushiriki_policy_engine_library.ICLR23ChallengeEnvironment:Model1Env_batchbeta',
)

register(
    id='ICLR23Workshop-env6-v0',
    entry_point='ushiriki_policy_engine_library.ICLR23ChallengeEnvironment:Model1Env_betalist',
)
