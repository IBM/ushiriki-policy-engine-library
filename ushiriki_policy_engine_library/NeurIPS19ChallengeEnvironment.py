'''
    Copyright 2019 IBM Corporation
    
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

class ChallengeEnvironment():
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

    def __init__(self, baseuri, experimentCount = 105, userID = "ChallengeUser", type = "", timeout = 0, realworkercount = 1, token=None):
        """
            The constructor for experiment class.
            
            Parameters:
            baseuri (string): The http endpoint for the task clerk.
            experimentCount (int): The experiment budget (the number of jobs permitted) for this experiment.
            userId (string): A valid userId able to create experiments for a given locationId.
            type (string): The variant of the base environment. May not be valid for all locations.
            timeout (int): The time interval in seconds that any job can be polled.
            realworkercount  (int): The number of jobs which can concurrently be active.
            
            Returns:
            None
            
            """
        self._realworkercount = realworkercount

        self.actionDimension = 2
        self.policyDimension = 5
        self._baseuri =  baseuri+"/neurips19/"+type if type is not "" else baseuri+"/neurips19"
        self.userId = userID
        self.token = token
        self._experimentCount = experimentCount
        self.experimentsRemaining = self._experimentCount
        self.history = []
        self.history1 = []
        self.reset()

    def reset(self):
        """Resets the state and clears all evidence of past actions."""
        self.state = 1
        self.done = False
        self.action = []
        ######
        self.labels = [0]
        self.rewards = [0]

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
            extended_action['state'] = self.state
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
            #print(policy)
            response = requests.post(rewardUrl, data = json.dumps(policy), headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'token':self.token, 'userID':self.userId});
            if response.status_code is not 200:
                raise ValueError("Environment Error. Check the baseuri and type. Also, checkout the suggestions related to the following error code: "+response.status_code)
            
            data = response.json();
            reward = float(data['data'])
            time.sleep(.05)
        except Exception as e:
            print(e);
            reward = float('nan')
        return reward

    def evaluateAction(self, action):
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
        print(self.experimentsRemaining, " Evaluations Remaining")
        if self.experimentsRemaining <= 0:
            raise ValueError('You have exceeded the permitted number of evaluations')

        if any([any((i<0, i>1)) for i in action]):
            raise ValueError('Interventions should be in [0,1]')
        try:
            action = [action[0],action[1]]
        except:
            raise ValueError('Two interventions are required per action')
        
        self.experimentsRemaining -= 1

        if ~self.done and self.state <= self.policyDimension:
            reward = self._simplePostAction(action)
            self.action = action
            self.state += 1

        if self.state > self.policyDimension: self.done = True
        self.history.append([self.state-1, action[0], action[1], reward])

        return self.state, reward, self.done, {}

    def evaluatePolicy(self, data, coverage = 1):
        """
            The function to post one or more policies and blocks until evaluated (or abandoned). Evaluations will be performed in parallel and are limited by the realworker count.
            
            Parameters:
            data (object): The policy or policies to be evaluated.
            coverage (float): The minimum portion of the data that must be returned.
            
            Raises:
            ValueError
            If Request would exceed the permitted number of Evaluations.
            If the policies to be evaluated aren't stored in a dictionary or a list of dictionaries.
            If any intervention is not in [0,1]
            
            Returns:
            reward: A list containing either the rewards associated with the provided action or nan if the action is not complete for all actions to be evaluated.
            
            """
        print(self.experimentsRemaining, " Evaluations Remaining")

        from multiprocessing import Pool
        if type(data) is list and all([type(i) is dict for i in data]): #list of policies
            for apolicy in data:
                if any([any((i[0]<0,i[0]>1,i[1]<0,i[1]>1)) for i in [apolicy[k] for k in apolicy]]):
                    raise ValueError('All interventions should be in [0,1]')
            self.experimentsRemaining -= len(data)*self.policyDimension
            if self.experimentsRemaining < 0:
                raise ValueError('Request would exceed the permitted number of Evaluations')
            pool = Pool(self._realworkercount)
            result = pool.map(self._simplePostPolicy, data)
            pool.close()
            pool.join()
            self.history1.append([i for i in zip(data,result)])
        elif type(data) is dict:
            self.experimentsRemaining -= 1*self.policyDimension
            if self.experimentsRemaining < 0:
                raise ValueError('Request would exceed the permitted number of Evaluations')
            result = self._simplePostPolicy(data)
            self.history1.append([data,result])
        else:
            raise ValueError('argument should be a policy (dictionary) or a list of policies')

        return result
