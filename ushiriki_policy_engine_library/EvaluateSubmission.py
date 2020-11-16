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
import pandas as pd
import numpy as np
import itertools

class EvaluateChallengeSubmission():
    """
        A class which defines how the Challenge submissions are to be evaluated.
        
        Attributes:
        environment (object): The environment used to assess the performence of the selected policies and actions.
        agent (object): The agent class which learns from the environment, and develops optimal (or at least good) policies/actions.
        episode_number (int): The number of episodes used during the training process. DO NOT CHANGE.
        
        """
    def __init__(self, environment, agent, filename = 'my_submission.csv', episode_number = 20):
        """
            The constructor for evaluation class.
            
            Parameters:
            environment (object): The environment used to assess the performence of the selected policies and actions.
            agent (object): The agent class which learns from the environment, and develops optimal (or at least good) policies/actions.
            filename (string): User selected file to save the assessment output.
            episode_number (int): The number of episodes used during the training process. DO NOT CHANGE.

            Returns:
            None
            
            """
        self.environment = environment
        self.agent = agent
        self.episode_number = episode_number
        self.reset();
        print(self.scoringFunction())
        self.create_submissions(filename)

    def reset(self):
        """Resets the state and clears all evidence of past actions."""
        self.policies = []
        self.rewards = []
        self.run = []

    def scoringFunction(self):
        """Initializes an instance of the environment, and an instance of the agent (with the new environment).
            Accesses the agent's `generate` method to begin the training process and generate the best policy.
            Repeats this process 10 times."""
        #Should be parallized
        for ii in range(10):
            e = self.environment()
            a = self.agent(e);
            finalpolicy, episodicreward = a.generate()
            self.policies.append(finalpolicy)
            self.rewards.append(episodicreward)
            self.run.append(ii)
        
        return np.median(self.rewards)

    def create_submissions(self, filename = 'my_submission.csv'):
        """
            Using data collected during the scoring function, populate a csv file with the policies learned.
            The scores for these policies are included as a convenience, but will neede to be recalculated.
            
            Parameters:
            filename (string): File to save the assessment output.
            
            """
        labels = ['run', 'reward', 'policy']
        rewards = np.array(self.rewards)
        data = { 'run': self.run,
            'rewards': rewards,
                'policy': self.policies,
                }
        submission_file = pd.DataFrame(data)
        submission_file.to_csv(filename, index=False)

class EvaluateAugmentedChallengeSubmission():
    """
        A class which defines how the Challenge submissions are to be evaluated.
        
        Attributes:
        environment (object): The environment used to assess the performence of the selected policies and actions.
        agent (object): The agent class which learns from the environment, and develops optimal (or at least good) policies/actions.
        episode_number (int): The number of episodes used during the training process. DO NOT CHANGE.
        
        """
    def __init__(self, environment, agent, filename = 'my_submission.csv', episode_number = 20):
        """
            The constructor for evaluation class.
            
            Parameters:
            environment (object): The environment used to assess the performence of the selected policies and actions.
            agent (object): The agent class which learns from the environment, and develops optimal (or at least good) policies/actions.
            filename (string): User selected file to save the assessment output.
            episode_number (int): The number of episodes used during the training process. DO NOT CHANGE.
            score(number): The score assigned to the evaluated submission.

            Returns:
            None
            
            """
        self.environment = environment
        self.agent = agent
        self.episode_number = episode_number
        self.reset();
        self.filename = filename
        print(self.scoringFunction())
        self.create_submissions()

    def reset(self):
        """Resets the state and clears all evidence of past actions."""
        self.policies = []
        self.rewards = []
        self.run = []
        self.allhistory = []
        self.score = None

    def scoringFunction(self):
        """Initializes an instance of the environment, and an instance of the agent (with the new environment).
            Accesses the agent's `generate` method to begin the training process and generate the best policy.
            Repeats this process 10 times."""
        #Should be parallized
        for ii in range(10):
            e = self.environment()
            a = self.agent(e);
            finalpolicy, episodicreward = a.generate()
            self.allhistory.append(e.history)
            self.allhistory.append(e.history1)
            self.policies.append(finalpolicy)
            self.rewards.append(episodicreward)
            self.run.append(ii)
        self.score = np.median(self.rewards)
        return self.score

    def create_submissions(self):
        """
            Using data collected during the scoring function, populate a csv file with the policies learned.
            The scores for these policies are included as a convenience, but will neede to be recalculated.
            
            Parameters:
            filename (string): File to save the assessment output.
            
            """
        from itertools import zip_longest
        flatten = lambda l: [item for sublist in l for item in sublist]
        rewards = np.array(self.rewards)
        data = { 'run': self.run,
            'rewards': rewards,
                'policy': self.policies,
                }
    
        if len(self.allhistory[0])==0:
            #if learning from full policies
            data2 = {'run':-1-np.arange(len(flatten(flatten(self.allhistory)))),
                    'rewards':[i[1] for j in range(1,len(self.allhistory)+1,2) for  i in self.allhistory[j][0]] ,
                    'policy':[i[0] for j in range(1,len(self.allhistory)+1,2) for  i in self.allhistory[j][0]]
                    }
        else:
            #if learning from sequences of actions
            dim = self.environment().policyDimension
            data2 = {'run':-1-np.arange(len(flatten(self.allhistory[::2]))/dim),
                'rewards':[sum([k[-1] for k in j]) for j in [i for i in zip_longest(*[iter(flatten(self.allhistory[::2]))]*dim)]],
                'policy':[{k[0]:[k[1],k[2]] for k in j} for j in [i[:dim] for i in zip_longest(*[iter(flatten(self.allhistory[::2]))]*dim)]]
                }
        submission_file = pd.concat([pd.DataFrame(data),pd.DataFrame(data2)] , ignore_index=True)
        submission_file.to_csv(self.filename, index=False)

class EvaluateAugmentedChallengeGymSubmission():
    """
    A class which defines how the Challenge submissions are to be evaluated.

    Attributes:
    environment (object): The environment used to assess the performence of the selected policies and actions.
    agent (object): The agent class which learns from the environment, and develops optimal (or at least good) policies/actions.
    episode_number (int): The number of episodes used during the training process. DO NOT CHANGE.

    """
    def __init__(self, environment, agent, filename = 'my_submission.csv', episode_number = 20):
        """
            The constructor for evaluation class.
            
            Parameters:
            environment (object): The environment used to assess the performence of the selected policies and actions.
            agent (object): The agent class which learns from the environment, and develops optimal (or at least good) policies/actions.
            filename (string): User selected file to save the assessment output.
            episode_number (int): The number of episodes used during the training process. DO NOT CHANGE.
            score(number): The score assigned to the evaluated submission.
            Returns:
            None
            
            """
        self.environment = environment
        self.agent = agent
        self.episode_number = episode_number
        self.reset();
        self.filename = filename
        print(self.scoringFunction())
        self.create_submissions()

    def reset(self):
        """Resets the state and clears all evidence of past actions."""
        self.policies = []
        self.rewards = []
        self.run = []
        self.allhistory = []
        self.score = None

    def scoringFunction(self, n = 10):
        import copy
        """Initializes an instance of the environment, and an instance of the agent (with the new environment).
            Accesses the agent's `generate` method to begin the training process and generate the best policy.
            Repeats this process n times."""
        
        for ii in range(n):
            e = copy.deepcopy(self.environment)
            a = self.agent(e);
            finalpolicy, episodicreward = a.generate()
            self.allhistory.append(e.env._env.history)
            self.allhistory.append(e.env._env.history1)
            self.policies.append(finalpolicy)
            self.rewards.append(episodicreward)
            self.run.append(ii)
        self.score = np.median(self.rewards)
        return self.score

    def create_submissions(self):
        """
            Using data collected during the scoring function, populate a csv file with the policies learned.
            The scores for these policies are included as a convenience, but will neede to be recalculated.
            
            Parameters:
            filename (string): File to save the assessment output.
            
            """
        from itertools import zip_longest
        flatten = lambda l: [item for sublist in l for item in sublist]
        rewards = np.array(self.rewards)
        data = { 'run': self.run,
            'rewards': rewards,
                'policy': self.policies,
                }
    
        if len(self.allhistory[0])==0:
            #if learning from full policies
            data2 = {'run':-1-np.arange(len(flatten(self.allhistory))),
                    'rewards':[i[1] for j in range(1,len(self.allhistory)+1,2) for  i in self.allhistory[j]] ,
                    'policy':[i[0] for j in range(1,len(self.allhistory)+1,2) for  i in self.allhistory[j]]
                    }
        else:
            #if learning from sequences of actions
            dim = len(self.environment.env.observation_space)
            data2 = {'run':-1-np.arange(len(flatten(self.allhistory[::2]))/dim),
                'rewards':[sum([k[-1] for k in j]) for j in [i for i in zip_longest(*[iter(flatten(self.allhistory[::2]))]*dim)]],
                'policy':[{k[0]:[k[1],k[2]] for k in j} for j in [i[:dim] for i in zip_longest(*[iter(flatten(self.allhistory[::2]))]*dim)]]
                }
        submission_file = pd.concat([pd.DataFrame(data),pd.DataFrame(data2)] , ignore_index=True)
        submission_file.to_csv(self.filename, index=False)

class EvaluateAugmentedGymSubmission():
    """
    A class which defines how the Challenge submissions are to be evaluated.

    Attributes:
    environment (object): The environment used to assess the performence of the selected policies and actions.
    agent (object): The agent class which learns from the environment, and develops optimal (or at least good) policies/actions.
    episode_number (int): The number of episodes used during the training process. DO NOT CHANGE.

    """
    def __init__(self, environment, agent, filename = 'my_submission.csv', episode_number = 20):
        """
            The constructor for evaluation class.
            
            Parameters:
            environment (object): The environment used to assess the performence of the selected policies and actions.
            agent (object): The agent class which learns from the environment, and develops optimal (or at least good) policies/actions.
            filename (string): User selected file to save the assessment output.
            episode_number (int): The number of episodes used during the training process. DO NOT CHANGE.
            score(number): The score assigned to the evaluated submission.
            Returns:
            None
            
            """
        self.environment = environment
        self.agent = agent
        self.episode_number = episode_number
        self.reset();
        self.filename = filename
        print(self.scoringFunction())
        self.create_submissions()

    def reset(self):
        """Resets the state and clears all evidence of past actions."""
        self.policies = []
        self.rewards = []
        self.run = []
        self.allstateactions = []
        self.allrewards = []
        self.score = None

    def scoringFunction(self, n = 10):
        import copy
        """Initializes an instance of the environment, and an instance of the agent (with the new environment).
            Accesses the agent's `generate` method to begin the training process and generate the best policy.
            Repeats this process n times."""
        
        for ii in range(n):
            e = copy.deepcopy(self.environment)
            a = self.agent(LoggerWrapper(e), self.episode_number);
            finalpolicy, episodicreward = a.generate()
            self.allrewards.append([np.sum([i[2] for i in episodes]) for episodes in a.environment.episodes])
            self.allstateactions.append([[i[0:2] for i in episodes] for episodes in a.environment.episodes])
            self.policies.append(finalpolicy)
            self.rewards.append(episodicreward)
            self.run.append(ii)
        self.score = np.median(self.rewards)
        return self.score

    def create_submissions(self):
        """
            Using data collected during the scoring function, populate a csv file with the policies learned.
            The scores for these policies are included as a convenience, but will neede to be recalculated.
            
            Parameters:
            filename (string): File to save the assessment output.
            
            """
        from itertools import zip_longest
        flatten = lambda l: [item for sublist in l for item in sublist]
        rewards = np.array(self.rewards)
        data = { 'run': self.run,
            'rewards': rewards,
                'policy': self.policies,
                }            
        if type(self.allrewards) is list and type(self.allrewards[0]) is list:
            #if learning from full policies
            data2={'run':-1-np.arange(len(list(itertools.chain(*self.allrewards)))),'rewards':list(itertools.chain(*self.allrewards)), 'policy':list(itertools.chain(*self.allstateactions))}
        else:
            #if learning from sequences of actions
            data2={'run':-1-np.arange(len(self.allrewards)),'rewards':self.allrewards, 'policy':self.allstateactions}   
        submission_file = pd.concat([pd.DataFrame(data),pd.DataFrame(data2)] , ignore_index=True)
        submission_file.to_csv(self.filename, index=False)
