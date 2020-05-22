
import gym
from ushiriki_policy_engine_library.SimpleChallengeEnvironment import ChallengeEnvironment

class ChallengeActionEnv(gym.Env):
    def __init__(self, baseuri="https://reward-service.eu-gb.mybluemix.net", userID="61122946-1832-11ea-ssss-github"):
        self._env = ChallengeEnvironment(baseuri=baseuri, userID=userID , experimentCount=2000)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        
    def reset(self):
        self._env.reset()
        return self._env.state
        
    def step(self, action):
        return self._env.evaluateAction(action)
