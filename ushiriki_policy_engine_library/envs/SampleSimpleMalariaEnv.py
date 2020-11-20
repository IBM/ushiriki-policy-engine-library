'''
    Copyright 2020 IBM Corporation
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
    http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    
    '''

import os
from sys import exit, exc_info, argv
import random
import json
import requests
import time
import gym
import numpy as np

class SimpleMalariaActionEnv(gym.Env):
    """
        This is the class which defines the simple Sequential Decision-making Environment object, and enables either actions or policies to be evaluated.
        
        Attributes:
        _resolution (string): The population size.
        _timeout (int): The time interval in seconds that any job can be polled.
        _realworkercount (int): The number of jobs which can concurrently be active.
        actionDimension (int): The number of elements in the action space.
        policyDimension (int): The number of elements in the policy.
        _baseuri (string): The http endpoint for the task clerk.
        userId (string): The unique user id.
        experimentsRemaining (int): The number of jobs that are remaining in the experiment budget.
        action (list): Temporary data storage for action.
        labels (list): Sequence of classes that an action can be categorized in for each state.
        rewards (list): Memory for past rewards which have been received.
        
        """

    def __init__(self, baseuri="http://reward-service.eu-gb.mybluemix.net/malariasurrogate", userID = "ChallengeUser", surrtype = "prove", timeout = 0, realworkercount = 1, token=None):
        """
            The constructor for experiment class.
            
            Parameters:
            baseuri (string): The http endpoint for the task clerk.
            userId (string): A valid userId able to create experiments for a given locationId.
            surrtype (string): The variant of the base environment. May not be valid for all locations.
            timeout (int): The time interval in seconds that any job can be polled.
            realworkercount  (int): The number of jobs which can concurrently be active.
            
            Returns:
            None
            
            """
        self._realworkercount = realworkercount

        self.actionDimension = 2
        self.policyDimension = 5
        self.statedata = ["CPDA"]
        baseuri = baseuri.strip("/")
        self._baseuri =  baseuri+"/"+surrtype if surrtype is not "" else baseuri
        self.userId = userID
        self.token = token
        self.action_space = gym.spaces.Box(low=np.array([0]*self.actionDimension, dtype=np.float64), high=np.array([1]*self.actionDimension, dtype=np.float64), dtype=np.float64)
        self.observation_space = gym.spaces.Discrete(self.policyDimension)
        self.history = []
        self.history1 = []
        self.reset()

    def reset(self):
        """Resets the state and clears all evidence of past actions."""
        self.state = 0
        self.done = False
        self.action = []
        ######
        self.labels = [0]
        self.rewards = [0]
        return self.state

    def _simplePostAction(self, action):
        """
            The helper function to get the reward for a given action.
            
            Parameters:
            action (object): The action to be posted.
            
            Returns:
            reward: The reward associated with the provided action or nan if the job is not complete.
            
            Raises:
            ValueError
            If response status is not 200.
            """
        rewardUrl = '%s/evaluate/action/'%self._baseuri

        try:
            #print(action)
            extended_action = {}
            extended_action['action']=action
            extended_action['old'] = self.action
            extended_action['state'] = self.state+1
            ##
            extended_action['labels'] = self.labels
            extended_action['rewards'] = self.rewards

            #print(extended_action)
            response = requests.post(rewardUrl, data = json.dumps(extended_action), headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'token':self.token, 'userID':self.userId});
            if response.status_code is not 200:
                raise ValueError("Invalid Environment. Check the baseuri and type.")
            
            data = response.json();
            
            reward = float(data['data'][0])
            self.rewards.append(reward)
            
            label = int(data['data'][1])
            self.labels.append(label)

        except Exception as e:
            print(e);
            reward = float('nan')
        return reward

    def step(self, action):
        """
            A `gym compliant` method to evaluate an action's utility.
            
            Parameters:
            action (object): The action to be posted.
            
            Raises:
            ValueError
            If Request would exceed the permitted number of Evaluations.
            If the interventions in the actions are not in [0,1].
            If there are not two interventions per action.
            
            Returns:
            unnamed (tuple): The next state, the provided reward, is the next state the terminal state, {}.
            
            """
        reward = float("nan")
        assert self.action_space.contains(action), "Invalid action: %s"%action
        assert not self.done, "The environment must be reset to continue"
        if not self.done and self.state <= self.policyDimension:
            reward = self._simplePostAction([action[0],action[1]])
            self.action = [action[0],action[1]]
            self.state += 1
            self.history.append([self.state-1, action[0], action[1], reward])
        if self.state >= self.policyDimension: self.done = True

        return self.state, reward, self.done, {}

class SimpleMalariaPolicyEnv(gym.Env):
    """
        This is the class which defines the simple Sequential Decision-making Environment object, and enables either actions or policies to be evaluated.
        
        Attributes:
        _resolution (string): The population size.
        _timeout (int): The time interval in seconds that any job can be polled.
        _realworkercount (int): The number of jobs which can concurrently be active.
        actionDimension (int): The number of elements in the action space.
        policyDimension (int): The number of elements in the policy.
        _baseuri (string): The http endpoint for the task clerk.
        userId (string): The unique user id.
        _experimentCount (int): The experiment budget.
        experimentsRemaining (int): The number of jobs that are remaining in the experiment budget.
        action (list): Temporary data storage for action.
        labels (list): Sequence of classes that an action can be categorized in for each state.
        rewards (list): Memory for past rewards which have been received.
        
        """

    def __init__(self, baseuri="http://reward-service.eu-gb.mybluemix.net/malariasurrogate", userID = "ChallengeUser", surrtype = "prove", timeout = 0, realworkercount = 1, token=None):
        """
            The constructor for experiment class.
            
            Parameters:
            baseuri (string): The http endpoint for the task clerk.
            experimentCount (int): The experiment budget (the number of jobs permitted) for this experiment.
            userId (string): A valid userId able to create experiments for a given locationId.
            surrtype (string): The variant of the base environment. May not be valid for all locations.
            timeout (int): The time interval in seconds that any job can be polled.
            realworkercount  (int): The number of jobs which can concurrently be active.
            
            Returns:
            None
            
            """
        self._realworkercount = realworkercount

        self.actionDimension = 2
        self.policyDimension = 5
        self.statedata = ["CPDA"]
        baseuri = baseuri.strip("/")
        self._baseuri =  baseuri+"/"+surrtype if surrtype is not "" else baseuri
        self.userId = userID
        self.token = token
        self.action_space = gym.spaces.Box(low=np.array([0]*self.actionDimension*self.policyDimension, dtype=np.float64), high=np.array([1]*self.actionDimension*self.policyDimension, dtype=np.float64), dtype=np.float64)
        self.observation_space = gym.spaces.MultiDiscrete([self.policyDimension]*self.policyDimension)
        self.history = []
        self.history1 = []
        self.reset()

    def reset(self):
        """Resets the state and clears all evidence of past actions."""
        self.state = [0,1,2,3,4]
        self.done = False
        self.action = []
        ######
        self.labels = [0]
        self.rewards = [0]

    def _simplePostPolicy(self, policy):
        """
            The helper function to get the reward for a given policy (sequence of actions).
            
            Parameters:
            policy (object): The policy to be posted.
            
            Returns:
            reward: The episodic reward associated with the provided policy or nan if the job is not complete.
            
            Raises:
            ValueError
            If response status is not 200.
            """
        rewardUrl = '%s/evaluate/policy/'%self._baseuri

        try:
            response = requests.post(rewardUrl, data = json.dumps(policy), headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'token':self.token, 'userID':self.userId});
            if response.status_code is not 200:
                raise ValueError("Environment Error. Check the baseuri and type. Also, checkout the suggestions related to the following error code: "+response.status_code)
            
            data = response.json();
            reward = float(data['data'])
        except Exception as e:
            print(e);
            reward = float('nan')
        return reward

    def step(self, policy):
        """
            A `gym compliant` method to evaluate an action's utility.
            
            Parameters:
            policy (object): The policy to be posted.
            
            Raises:
            ValueError
            If the interventions in the actions are not in the action space.
            
            Returns:
            unnamed (tuple): The next state, the provided reward, is the next state the terminal state, {}.
            
            """
        policy = np.array(policy) #forcing it to be an nparray
        assert self.action_space.contains(policy), "Invalid action: %s"%policy
        data = {"1":policy[0:2].tolist(),"2":policy[2:4].tolist(),"3":policy[4:6].tolist(),"4":policy[6:8].tolist(),"5":policy[8:10].tolist()
        }
        result = self._simplePostPolicy(data)
        self.action = policy
        self.done = True
        self.history1.append([data,result])
        reward = result
        return self.state, reward, self.done, {}

class Box2DToDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, width = 10):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), \
            "Should only be used to wrap Continuous envs."
        
        self.width = width
        self.action_space = gym.spaces.Discrete(np.power(self.width, env.action_space.shape[0]))

    def action(self, action):
        row    = (int)(action / self.width)
        column = action % self.width
        return np.array([row/self.width, column/self.width])
