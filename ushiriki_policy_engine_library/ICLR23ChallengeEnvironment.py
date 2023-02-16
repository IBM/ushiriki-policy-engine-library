'''
    Copyright 2023 IBM Corporation
    
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

import gym
import numpy as np
import pandas as pd
import samplemetamodel.langata as model0
import covasimmetamodels.langata as model1
#import hpvsimmetamodels.langata as model2

class Model0Env_betalist(gym.Env):
    def __init__(self, model_name="https://127.0.0.1:8080/modelbasedata/json/langatacovid19modelv1/",
                 userID="61122946-1832-11ea-ssss-github", driver_data="casesdata.csv",
                 numdays = 14, duration = 182, startpt=0, maxpop = 100000000.0, token = None):
        #seirdv ignoring ev
        self.token = token
        self.userID = userID
        self.window = numdays
        self.max_pop = maxpop
        self.num_windows = int(duration/self.window)
        self.duration = self.num_windows*self.window

        self.action_space = gym.spaces.Box(
            low=np.array([.001]*self.num_windows+[0.2]+[0.000]+[0.0]), 
            high=np.array([.3]*self.num_windows+[1]+[0.003]+[.05]), dtype=float)
        self.observation_space = gym.spaces.Box(low=0, high=self.max_pop, shape=(5,), dtype=float)

        self.parms = pd.read_json(model_name)
        casedata_ = pd.read_csv(driver_data)

        self.N = casedata_['population'].tolist()[startpt]
        self.output0 = casedata_['confirmed_cases'].tolist()[startpt:]
        self.output1 = casedata_['deaths'].tolist()[startpt:]
        self.day0 = casedata_['dt'].tolist()[startpt]

        assert self.duration <= len(self.output0), "Output0 length does not match the length of the model driver"
        assert self.duration <= len(self.output1), "Output1 length does not match the length of the model driver"
        self.R0 = 0
        self.E0 = self.parms["exposed"][0] if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]) else 0
        self.D0 = self.output1[0]
        self.I0 = self.output0[0]-self.D0-self.R0
        self.V0 = 0
        if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]):
            self.S0 = self.parms["susceptible"][0]
            self.parms["population"] = self.S0 + self.I0 + self.R0 + self.D0
        else:
            self.S0 = self.N - self.I0 - self.R0 - self.D0
            self.parms.at[0,"susceptible"] = self.S0
            self.parms["population"] = self.N

        self.parms["days"] = self.duration
        self.parms["day0"] = self.day0
        self.parms["infectious"] = self.I0
        self.parms["recovered"] = self.R0
        self.parms["deaths"] = self.D0

        self.parms["exposed"] = self.E0
        self.parms["vaccinated"] = self.V0
        self.statedata = ["ds", "di", "dr", "dd", "act"]
        self.actiondata = ["beta"+str(i) for i in range(self.num_windows)]
        self.reset()
        return

    def reset(self, seed=None, options=None):
        self.states = []
        self.actions = []
        self.tempactions = []
        self.jobids = []
        self.rewards = []
        return np.array([self.S0, self.I0, self.R0, self.D0, 0]), {}

    def step(self, action):
        done = False
        reward = None
        assert self.action_space.contains(action), "Invalid action: %s"%action
        if len(self.states) <= self.duration:
            self.actions.append(action)
            self.tempactions = np.repeat(action[0:-3], self.window)
            tmp = {}
            for ind in self.parms.keys():
                if ind == "beta":
                    tmp["beta"] = self.tempactions.tolist()
                elif ind == "alpha":
                    tmp["alpha"] = action[-3]
                elif ind == "d0":
                    tmp["d0"] = action[-2]
                elif ind == "gamma":
                    tmp["gamma"] = action[-1]
                else:
                    tmp[ind] = self.parms[ind][0]
            tmp["beta_window"]=self.window
            try:
                results = model0.run_model(tmp)
            except Exception as e:
                print(e)
                print(self.tempactions)
            self.states = np.array([[i['susceptible'],i['infectious'],i['recovered'],i['deaths']] for i in results])


            model_data = np.array([self.states[:,1:].sum(axis=1), self.states[:,3]]).T
            real_data = np.array([np.array(self.output0[:self.duration]), self.output1[:self.duration]]).T

            se = (real_data-model_data)**2
            reward = - np.mean(np.sqrt(np.mean(se, axis=0))/(np.amax(real_data, axis=0)-np.amin(real_data, axis=0)))
            # reward = - np.mean(np.sqrt(np.mean(se, axis=0)))
            self.rewards.append(reward)
        done = True
        state = np.hstack((self.states[-1],[self.num_windows]))
        return state, reward, done, False, {}
    
    
class Model0Env_batchbeta(gym.Env):
    def __init__(self, model_name="https://127.0.0.1:8080/modelbasedata/json/langatacovid19modelv1/",
                 userID="61122946-1832-11ea-ssss-github", driver_data="casesdata.csv",
                 numdays = 14, duration = 182, startpt=0, maxpop = 100000000.0, token = None):
        #seirdv ignoring ev
        
        self.token = token
        self.userID = userID
        self.window = numdays
        self.max_pop = maxpop
        self.num_windows = int(duration/self.window)
        self.duration = self.num_windows*self.window
        self.action_space = gym.spaces.Box(
            low=np.array([.001]*self.num_windows), 
            high=np.array([.3]*self.num_windows), dtype=float)
        self.observation_space = gym.spaces.Box(
            low=np.array([0]*4+[0]), 
            high=np.array([self.max_pop]*4 + [self.num_windows]), 
            dtype=float)
        self.parms = pd.read_json(model_name)
        casedata_ = pd.read_csv(driver_data)
        self.N = casedata_['population'].tolist()[startpt]
        self.output0 = casedata_['confirmed_cases'].tolist()[startpt:]
        self.output1 = casedata_['deaths'].tolist()[startpt:]
        self.day0 = casedata_['dt'].tolist()[startpt]
        assert self.duration <= len(self.output0), "Output0 length does not match the length of the model driver"
        assert self.duration <= len(self.output1), "Output1 length does not match the length of the model driver"
        self.R0 = 0
        self.E0 = self.parms["exposed"][0] if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]) else 0
        self.D0 = self.output1[0]
        self.I0 = self.output0[0]-self.D0-self.R0
        self.V0 = 0
        if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]):
            self.S0 = self.parms["susceptible"][0]
            self.parms["population"] = self.S0 + self.I0 + self.R0 + self.D0
        else:
            self.S0 = self.N - self.I0 - self.R0 - self.D0
            self.parms.at[0,"susceptible"] = self.S0
            self.parms["population"] = self.N
        # self.parms["days"] = self.duration
        self.parms["day0"] = self.day0
        self.parms["infectious"] = self.I0
        self.parms["recovered"] = self.R0
        self.parms["deaths"] = self.D0
        self.parms["exposed"] = self.E0
        self.parms["vaccinated"] = self.V0
        self.statedata = ["ds", "di", "dr", "dd", "window"]
        self.actiondata = ["beta"+str(i) for i in range(self.num_windows)]
        self.reset()
        return
    def reset(self, seed=None, options=None):
        self.states = []
        self.actions = []
        self.tempactions = []
        self.jobids = []
        self.rewards = []
        return np.array([self.S0, self.I0, self.R0, self.D0, 0]), {}
    def step(self, batchbeta):
        reward = None
        assert self.states == [] or self.states.shape[0] < self.duration, "Reset before providing another action"
        assert self.action_space.contains(batchbeta), "Invalid action: %s"%(batchbeta)
        self.actions.append(batchbeta)
        self.tempactions = np.repeat(batchbeta, self.window)
        tmp = {}
        for ind in self.parms.keys():
            if ind == "beta":
                tmp[ind] = self.tempactions.tolist()
            elif ind == "days":
                tmp[ind] = len(tmp["beta"])
            else:
                tmp[ind] = self.parms[ind][0]
        tmp["beta_window"]=self.window
        try:
            results = model0.run_model(tmp)
        except Exception as e:
            print(e)
            print(self.tempactions)
        self.states = np.array([[i['susceptible'],i['infectious'],i['recovered'],i['deaths']] for i in results])
        model_data = np.array([self.states[:,1:].sum(axis=1), self.states[:,3]]).T
        real_data = np.array([np.array(self.output0[:self.duration]), self.output1[:self.duration]]).T
        se = (real_data-model_data)**2
        se = np.array(np.split(se,self.num_windows))
        reward = - np.mean(np.sqrt(np.mean(se, axis=1))/(np.amax(real_data, axis=0)-np.amin(real_data, axis=0)+.00001),axis=1)
        self.rewards.append(reward)
        done = True
        state = np.append(self.states[::int(self.states.shape[0]/self.num_windows),:],np.arange(self.num_windows).reshape(-1,1),axis=1)
        return state, reward, done, False, {}
    

class Model0Env_beta(gym.Env):
    def __init__(self, model_name="https://127.0.0.1:8080/modelbasedata/json/langatacovid19modelv1/",
                 userID="61122946-1832-11ea-ssss-github", driver_data="casesdata.csv",
                 numdays = 14, duration = 182, startpt=0, maxpop = 100000000.0, token = None):
        #seirdv ignoring ev
        self.token = token
        self.userID = userID
        self.window = numdays
        self.max_pop = maxpop
        self.num_windows = int(duration/self.window)
        self.duration = self.num_windows*self.window

        self.action_space = gym.spaces.Box(
            low=np.array([.001]), 
            high=np.array([.3]), dtype=float)
        self.observation_space = gym.spaces.Box(
            low=np.array([0]*4+[0]), 
            high=np.array([self.max_pop]*4 + [self.num_windows]), 
            dtype=float)

        self.parms = pd.read_json(model_name)
        casedata_ = pd.read_csv(driver_data)

        self.N = casedata_['population'].tolist()[startpt]
        self.output0 = casedata_['confirmed_cases'].tolist()[startpt:]
        self.output1 = casedata_['deaths'].tolist()[startpt:]
        self.day0 = casedata_['dt'].tolist()[startpt]

        assert self.duration <= len(self.output0), "Output0 length does not match the length of the model driver"
        assert self.duration <= len(self.output1), "Output1 length does not match the length of the model driver"
        self.R0 = 0
        self.E0 = self.parms["exposed"][0] if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]) else 0
        self.D0 = self.output1[0]
        self.I0 = self.output0[0]-self.D0-self.R0
        self.V0 = 0
        if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]):
            self.S0 = self.parms["susceptible"][0]
            self.parms["population"] = self.S0 + self.I0 + self.R0 + self.D0
        else:
            self.S0 = self.N - self.I0 - self.R0 - self.D0
            self.parms.at[0,"susceptible"] = self.S0
            self.parms["population"] = self.N

        # self.parms["days"] = self.duration
        self.parms["day0"] = self.day0
        self.parms["infectious"] = self.I0
        self.parms["recovered"] = self.R0
        self.parms["deaths"] = self.D0

        self.parms["exposed"] = self.E0
        self.parms["vaccinated"] = self.V0
        self.statedata = ["ds", "di", "dr", "dd", "window"]
        self.actiondata = ["beta"+str(i) for i in range(self.num_windows)]
        self.reset()
        return

    def reset(self, seed=None, options=None):
        self.states = []
        self.actions = []
        self.tempactions = []
        self.jobids = []
        self.rewards = []
        return np.array([self.S0, self.I0, self.R0, self.D0, 0]), {}

    def step(self, beta):
        reward = None
        assert self.states == [] or self.states.shape[0] < self.duration, "Reset before providing another action"
        assert self.action_space.contains(beta), "Invalid action: %s"%beta
        self.actions.append(beta)
        self.tempactions = np.repeat(self.actions, self.window)
        tmp = {}
        for ind in self.parms.keys():
            if ind == "beta":
                tmp[ind] = self.tempactions.tolist()
            elif ind == "days":
                tmp[ind] = len(tmp["beta"])
            else:
                tmp[ind] = self.parms[ind][0]
        tmp["beta_window"]=self.window
        try:
            results = model0.run_model(tmp)
        except Exception as e:
            print(e)
            print(self.tempactions)
        self.states = np.array([[i['susceptible'],i['infectious'],i['recovered'],i['deaths']] for i in results])


        model_data = np.array([self.states[-self.window+len(tmp["beta"]):len(tmp["beta"]),1:].sum(axis=1), self.states[-self.window+len(tmp["beta"]):len(tmp["beta"]),3]]).T
        real_data = np.array([np.array(self.output0[-self.window+len(tmp["beta"]):len(tmp["beta"])]), self.output1[-self.window+len(tmp["beta"]):len(tmp["beta"])]]).T

        se = (real_data-model_data)**2
        reward = - np.mean(np.sqrt(np.mean(se, axis=0))/(np.amax(real_data, axis=0)-np.amin(real_data, axis=0)+.00001))
        self.rewards.append(reward)

        if len(self.states) < self.duration:
            done = False
        else:
            done = True
        state = np.append(self.states[-1],len(self.actions)-1)

        return state, reward, done, False, {}


class Model1Env_betalist(gym.Env):
    def __init__(self, model_name="https://127.0.0.1:8080/modelbasedata/json/langatacovid19modelv1/",
                 userID="61122946-1832-11ea-ssss-github", driver_data="casesdata.csv",
                 numdays = 14, duration = 182, startpt=0, maxpop = 100000000.0, token = None):
        #seirdv ignoring ev
        self.token = token
        self.userID = userID
        self.window = numdays
        self.max_pop = maxpop
        self.num_windows = int(duration/self.window)
        self.duration = self.num_windows*self.window

        self.action_space = gym.spaces.Box(
            low=np.array([.001]*self.num_windows), 
            high=np.array([.05]*self.num_windows), dtype=float)
        self.observation_space = gym.spaces.Box(low=0, high=self.max_pop, shape=(5,), dtype=float)

        self.parms = pd.read_json(model_name)
        casedata_ = pd.read_csv(driver_data)

        self.N = casedata_['population'].tolist()[startpt]
        self.output0 = casedata_['confirmed_cases'].tolist()[startpt:]
        self.output1 = casedata_['deaths'].tolist()[startpt:]
        self.day0 = casedata_['dt'].tolist()[startpt]

        assert self.duration <= len(self.output0), "Output0 length does not match the length of the model driver"
        assert self.duration <= len(self.output1), "Output1 length does not match the length of the model driver"
        self.R0 = 0
        self.E0 = self.parms["exposed"][0] if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]) else 0
        self.D0 = self.output1[0]
        self.I0 = self.output0[0]-self.D0-self.R0
        self.V0 = 0
        if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]):
            self.S0 = self.parms["susceptible"][0]
            self.parms["population"] = self.S0 + self.I0 + self.R0 + self.D0
        else:
            self.S0 = self.N - self.I0 - self.R0 - self.D0
            self.parms.at[0,"susceptible"] = self.S0
            self.parms["population"] = self.N

        self.parms["days"] = self.duration
        self.parms["day0"] = self.day0
        self.parms["infectious"] = self.I0
        self.parms["recovered"] = self.R0
        self.parms["deaths"] = self.D0

        self.parms["exposed"] = self.E0
        self.parms["vaccinated"] = self.V0
        self.statedata = ["ds", "di", "dr", "dd", "act"]
        self.actiondata = ["beta"+str(i) for i in range(self.num_windows)]
        self.reset()
        return

    def reset(self, seed=None, options=None):
        self.states = []
        self.actions = []
        self.tempactions = []
        self.jobids = []
        self.rewards = []
        return np.array([self.S0, self.I0, self.R0, self.D0, 0]), {}

    def step(self, action):
        done = False
        reward = None
        assert self.action_space.contains(action), "Invalid action: %s"%action
        if len(self.states) <= self.duration:
            self.actions.append(action)
            self.tempactions = np.array(action).flatten()
            tmp = {}
            for ind in self.parms.keys():
                if ind == "beta":
                    #calculate the relative changes in beta to use with cv.change_beta
                    tmp["beta"] = np.concatenate(([1],self.tempactions[1:]/self.tempactions[:-1])).tolist()
                elif ind == "d0":
                    tmp["d0"] = 1
                elif ind == "beta0":
                    tmp["beta0"] = action[0]
                else:
                    tmp[ind] = self.parms[ind][0]
            tmp["beta_window"]=self.window
            try:
                results = model1.run_model(tmp)
            except Exception as e:
                print(e)
                print(self.tempactions)
            self.states = np.array([[i['susceptible'],i['infectious'],i['recovered'],i['deaths']] for i in results])


            model_data = np.array([self.states[:,1:].sum(axis=1), self.states[:,3]]).T
            real_data = np.array([np.array(self.output0[:self.duration]), self.output1[:self.duration]]).T

            se = (real_data-model_data)**2
            reward = - np.mean(np.sqrt(np.mean(se, axis=0))/(np.amax(real_data, axis=0)-np.amin(real_data, axis=0)))
            # reward = - np.mean(np.sqrt(np.mean(se, axis=0)))
            self.rewards.append(reward)
        done = True
        state = np.hstack((self.states[-1],[self.num_windows]))
        return state, reward, done, False, {}
    
    
class Model1Env_batchbeta(gym.Env):
    def __init__(self, model_name="https://127.0.0.1:8080/modelbasedata/json/langatacovid19modelv1/",
                 userID="61122946-1832-11ea-ssss-github", driver_data="casesdata.csv",
                 numdays = 14, duration = 182, startpt=0, maxpop = 100000000.0, token = None):
        #seirdv ignoring ev
        
        self.token = token
        self.userID = userID
        self.window = numdays
        self.max_pop = maxpop
        self.num_windows = int(duration/self.window)
        self.duration = self.num_windows*self.window
        self.action_space = gym.spaces.Box(
            low=np.array([.001]*self.num_windows), 
            high=np.array([.05]*self.num_windows), dtype=float)
        self.observation_space = gym.spaces.Box(
            low=np.array([0]*4+[0]), 
            high=np.array([self.max_pop]*4 + [self.num_windows]), 
            dtype=float)
        self.parms = pd.read_json(model_name)
        casedata_ = pd.read_csv(driver_data)
        self.N = casedata_['population'].tolist()[startpt]
        self.output0 = casedata_['confirmed_cases'].tolist()[startpt:]
        self.output1 = casedata_['deaths'].tolist()[startpt:]
        self.day0 = casedata_['dt'].tolist()[startpt]
        assert self.duration <= len(self.output0), "Output0 length does not match the length of the model driver"
        assert self.duration <= len(self.output1), "Output1 length does not match the length of the model driver"
        self.R0 = 0
        self.E0 = self.parms["exposed"][0] if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]) else 0
        self.D0 = self.output1[0]
        self.I0 = self.output0[0]-self.D0-self.R0
        self.V0 = 0
        if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]):
            self.S0 = self.parms["susceptible"][0]
            self.parms["population"] = self.S0 + self.I0 + self.R0 + self.D0
        else:
            self.S0 = self.N - self.I0 - self.R0 - self.D0
            self.parms.at[0,"susceptible"] = self.S0
            self.parms["population"] = self.N
        self.parms["days"] = self.duration
        self.parms["day0"] = self.day0
        self.parms["infectious"] = self.I0
        self.parms["recovered"] = self.R0
        self.parms["deaths"] = self.D0
        self.parms["exposed"] = self.E0
        self.parms["vaccinated"] = self.V0
        self.statedata = ["ds", "di", "dr", "dd", "window"]
        self.actiondata = ["beta"+str(i) for i in range(self.num_windows)]
        self.reset()
        return
    def reset(self, seed=None, options=None):
        self.states = []
        self.actions = []
        self.tempactions = []
        self.jobids = []
        self.rewards = []
        return np.array([self.S0, self.I0, self.R0, self.D0, 0]), {}
    def step(self, batchbeta):
        reward = None
        assert self.states == [] or self.states.shape[0] < self.duration, "Reset before providing another action"
        assert self.action_space.contains(batchbeta), "Invalid action: %s"%(batchbeta)

        self.actions.append(batchbeta)
        self.tempactions = batchbeta.flatten()
        tmp = {}
        for ind in self.parms.keys():
            if ind == "beta":
                #calculate the relative changes in beta to use with cv.change_beta
                tmp["beta"] = np.concatenate(([1],self.tempactions[1:]/self.tempactions[:-1])).tolist()
            elif ind == "d0":
                tmp["d0"] = 1
            elif ind == "beta0":
                tmp["beta0"] = batchbeta[0]
            else:
                tmp[ind] = self.parms[ind][0]
        tmp["beta_window"]=self.window
        try:
            results = model1.run_model(tmp)
        except Exception as e:
            print(e)
            print(self.tempactions)
        self.states = np.array([[i['susceptible'],i['infectious'],i['recovered'],i['deaths']] for i in results])
        model_data = np.array([self.states[:,1:].sum(axis=1), self.states[:,3]]).T
        real_data = np.array([np.array(self.output0[:self.duration]), self.output1[:self.duration]]).T
        se = (real_data-model_data)**2
        se = np.array(np.split(se,self.num_windows))
        reward = - np.mean(np.sqrt(np.mean(se, axis=1))/(np.amax(real_data, axis=0)-np.amin(real_data, axis=0)+.00001),axis=1)
        self.rewards.append(reward)
        done = True
        state = np.append(self.states[::int(self.states.shape[0]/self.num_windows),:],np.arange(self.num_windows).reshape(-1,1),axis=1)
        return state, reward, done, False, {}
    

class Model1Env_beta(gym.Env):
    def __init__(self, model_name="https://127.0.0.1:8080/modelbasedata/json/langatacovid19modelv1/",
                 userID="61122946-1832-11ea-ssss-github", driver_data="casesdata.csv",
                 numdays = 14, duration = 182, startpt=0, maxpop = 100000000.0, token = None):
        #seirdv ignoring ev
        self.token = token
        self.userID = userID
        self.window = numdays
        self.max_pop = maxpop
        self.num_windows = int(duration/self.window)
        self.duration = self.num_windows*self.window

        self.action_space = gym.spaces.Box(
            low=np.array([.001]), 
            high=np.array([.05]), dtype=float)
        self.observation_space = gym.spaces.Box(
            low=np.array([0]*4+[0]), 
            high=np.array([self.max_pop]*4 + [self.num_windows]), 
            dtype=float)

        self.parms = pd.read_json(model_name)
        casedata_ = pd.read_csv(driver_data)

        self.N = casedata_['population'].tolist()[startpt]
        self.output0 = casedata_['confirmed_cases'].tolist()[startpt:]
        self.output1 = casedata_['deaths'].tolist()[startpt:]
        self.day0 = casedata_['dt'].tolist()[startpt]

        assert self.duration <= len(self.output0), "Output0 length does not match the length of the model driver"
        assert self.duration <= len(self.output1), "Output1 length does not match the length of the model driver"
        self.R0 = 0
        self.E0 = self.parms["exposed"][0] if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]) else 0
        self.D0 = self.output1[0]
        self.I0 = self.output0[0]-self.D0-self.R0
        self.V0 = 0
        if "susceptible" in self.parms.keys() and not np.isnan(self.parms["susceptible"][0]):
            self.S0 = self.parms["susceptible"][0]
            self.parms["population"] = self.S0 + self.I0 + self.R0 + self.D0
        else:
            self.S0 = self.N - self.I0 - self.R0 - self.D0
            self.parms.at[0,"susceptible"] = self.S0
            self.parms["population"] = self.N

        # self.parms["days"] = self.duration
        self.parms["day0"] = self.day0
        self.parms["infectious"] = self.I0
        self.parms["recovered"] = self.R0
        self.parms["deaths"] = self.D0

        self.parms["exposed"] = self.E0
        self.parms["vaccinated"] = self.V0
        self.statedata = ["ds", "di", "dr", "dd", "window"]
        self.actiondata = ["beta"+str(i) for i in range(self.num_windows)]
        self.reset()
        return

    def reset(self, seed=None, options=None):
        self.states = []
        self.actions = []
        self.tempactions = []
        self.jobids = []
        self.rewards = []
        return np.array([self.S0, self.I0, self.R0, self.D0, 0]), {}

    def step(self, beta):
        reward = None
        assert self.states == [] or self.states.shape[0] < self.duration, "Reset before providing another action"
        assert self.action_space.contains(beta), "Invalid action: %s"%beta
        self.actions.append(beta)
        self.tempactions = np.array(self.actions).flatten()
        tmp = {}
        for ind in self.parms.keys():
            if ind == "beta":
                #calculate the relative changes in beta to use with cv.change_beta
                tmp["beta"] = np.concatenate(([1],self.tempactions[1:]/self.tempactions[:-1])).tolist()
            elif ind == "d0":
                tmp["d0"] = 1
            elif ind == "beta0":
                tmp["beta0"] = self.tempactions[0]
            elif ind == "days":
                tmp[ind] = self.tempactions.shape[0]*self.window
            else:
                tmp[ind] = self.parms[ind][0]
        tmp["beta_window"]=self.window
        try:
            results = model1.run_model(tmp)
        except Exception as e:
            print(e)
            print(self.tempactions)
        self.states = np.array([[i['susceptible'],i['infectious'],i['recovered'],i['deaths']] for i in results])


        model_data = np.array([self.states[-self.window:,1:].sum(axis=1), self.states[-self.window:,3]]).T
        real_data = np.array([np.array(self.output0[-self.window:]), self.output1[-self.window:]]).T

        se = (real_data-model_data)**2
        reward = - np.mean(np.sqrt(np.mean(se, axis=0))/(np.amax(real_data, axis=0)-np.amin(real_data, axis=0)+.00001))
        self.rewards.append(reward)

        if len(self.states) < self.duration:
            done = False
        else:
            done = True
        state = np.append(self.states[-1],len(self.actions)-1)

        return state, reward, done, False, {}

