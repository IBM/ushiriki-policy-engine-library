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
