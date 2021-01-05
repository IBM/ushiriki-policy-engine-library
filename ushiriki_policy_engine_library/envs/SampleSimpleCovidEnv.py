import gym
import requests
import pandas as pd
import numpy as np

class CovidChallengeCalibrationEnv(gym.Env):
    def __init__(self, baseuri="https://reward-service.eu-gb.mybluemix.net/covid19modelv1", userID="61122946-1832-11ea-ssss-github", base_data="calibration_input.json", driver_data="casesdata.csv", numdays = 14, duration = 180, maxpop = 10000000.0, low=[0,0,0], high=[1,1,1], token = None):
        self.uri = baseuri.strip("/")+"/evaluate/policy/"
        self.token = token
        self.userID = userID
        self.statedata = ["ds", "di", "dr", "dd"]
        self.window = numdays
        self.max_pop = maxpop
        self.duration = duration
        self.action_space = gym.spaces.Box(low=gym.spaces.utils.np.array(low), high=gym.spaces.utils.np.array(high), dtype=float)
        self.observation_space = gym.spaces.Box(low=0, high=self.max_pop, shape=(len(self.statedata),), dtype=float)
        self.parms = pd.read_json(base_data)
        self.N =  self.parms["population"]
        casedata_ = pd.read_csv(driver_data)
        self.output0 = casedata_['active_cases'].tolist()
        self.output1 = casedata_['deaths'].tolist()
        self.output2 = casedata_['recovered'].tolist()
        if "susceptible" in casedata_.keys(): 
            self.output3 = casedata_['susceptible'].tolist()
            self.parms.at[0,"susceptible"] = self.output3[0] 
            
        self.parms["Deltap"]= [casedata_['100-wits_si'].tolist() for _ in self.parms.index]
        assert len(self.parms["Deltap"][0]) == len(self.output0), "Output0 length does not match the length of the model driver"
        assert len(self.parms["Deltap"][0]) == len(self.output1), "Output1 length does not match the length of the model driver"
        assert len(self.parms["Deltap"][0]) == len(self.output2), "Output2 length does not match the length of the model driver"
        self.I0 = casedata_['active_cases'].tolist()[0]
        self.R0 = self.output2[0]
        self.D0 = self.output1[0]
        self.S0 = self.parms["susceptible"] if np.isfinite(self.parms["susceptible"][0]) else self.N - self.I0 - self.R0 - self.D0
        self.parms["days"] = len(self.parms["Deltap"][0])
        self.parms["infectious"] = self.I0
        self.parms["deaths"] = self.D0
        self.parms["recovered"] = self.R0
        self.reset()
        return
        
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        return [self.S0, self.I0, self.R0, self.D0] #should probably clip to self.max_pop
        
    def step(self, action):
        reward = None
        assert self.action_space.contains(action), "Invalid action: %s"%action
        if len(self.states) <= self.duration:
            self.actions.append(action)
            self.parms["alphabeta"] = action[0]
            self.parms["gamma"] = action[1]
            self.parms["d0"] = action[2]
            response = requests.post(self.uri, data= str({i:self.parms[i][0] for i in self.parms}).replace("'","\"").replace("nan", "null"), headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'token':self.token, 'userID':self.userID})
            results = response.json()['data']
            self.states = gym.spaces.utils.np.array([[i['susceptible'],i['infectious'],i['recovered'],i['deaths']] for i in results])
            numcases = self.states[:,1]
            numdeaths = self.states[:,3]
            numrecovered = self.states[:,2]
            output0 = gym.spaces.utils.np.array(self.output0)
            output1 = gym.spaces.utils.np.array(self.output1)
            output2 = gym.spaces.utils.np.array(self.output2)
            subtr_cases=numcases-output0
            subtr_deaths=numdeaths-output1
            subtr_recovered=numrecovered-output2
            reward = - gym.spaces.utils.np.sqrt(gym.spaces.utils.np.sum(subtr_cases*subtr_cases.T) +gym.spaces.utils.np.sum(subtr_deaths*subtr_deaths.T) +gym.spaces.utils.np.sum(subtr_recovered*subtr_recovered.T))
            self.rewards.append(reward)

        done = True
        state = self.states[-1]

        return state, reward, done, {}


class CovidChallengeActionEnv(gym.Env):
    def __init__(self, baseuri="https://reward-service.eu-gb.mybluemix.net/covid19modelv1", userID="61122946-1832-11ea-ssss-github", base_data="https://gist.githubusercontent.com/slremy/fad09f9224a885fa460415cb4940fbd6/raw/dc48f0be9e94e5d277b47f8a42008e903547a405/birnin_zana.json", numdays = 14, duration = 180, maxpop = 10000000.0, low=[0,0,0], high=[1,1,1], token = None):
        self.uri = baseuri.strip("/")+"/evaluate/policy/"
        self.token = token
        self.userID = userID
        self.statedata = ["ds", "di", "dr", "dd"]
        self.window = numdays
        self.max_pop = maxpop
        self.duration = duration
        self.action_space = gym.spaces.Discrete(100)
        self.observation_space = gym.spaces.Box(low=0, high=self.max_pop, shape=(len(self.statedata),), dtype=float)
        self.parms = pd.read_json(base_data)
        self.N =  self.parms["population"][0]
        self.I0 = self.parms["infectious"][0] if pd.isna(self.parms["infectious"][0]) is False else 0
        self.R0 = self.parms["recovered"][0] if pd.isna(self.parms["recovered"][0]) is False else 0
        self.D0 = self.parms["deaths"][0] if pd.isna(self.parms["deaths"][0]) is False else 0
        self.S0 = self.parms["susceptible"][0] if pd.isna(self.parms["susceptible"][0]) is False else self.N - self.I0 - self.R0 - self.D0
        self.reset()
        return
        
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.tempactions = []
        return [self.S0, self.I0, self.R0, self.D0]

    def step(self, action):
        done = False
        reward = None
        d_index = None
        daly_no_int = [1464.9297629786774, 9273.788068402077, 92408.22192626673, 150539.6266037019, 120077.84028811468, 88207.69033717741, 64280.071878892995, 46782.87922640833, 34037.5707347189, 24761.85578834993, 18013.084872795735, 13103.380010518304, 9531.764237564494]
        assert self.action_space.contains(action), "Invalid action: %s"%action
        if len(self.states) <= self.duration:
            self.actions.append(action)
            for _ in range(self.window):
                self.tempactions.append(action)
            self.parms["Deltap"][0][:] = self.tempactions
            self.parms["days"] = len(self.parms["Deltap"][0])
            d_index = int(self.parms["days"]/self.window -1)

            response = requests.post(self.uri, data= str({i:self.parms[i][0] for i in self.parms}).replace("'","\"").replace("nan", "null"), headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'token':self.token, 'userID':self.userID})
            results = response.json()['data']
            self.states = gym.spaces.utils.np.array([[i['susceptible'],i['infectious'],i['recovered'],i['deaths']] for i in results])
            if len(self.rewards) == 0:
                # reward = - self.states[-1][1] - (99-action)*(self.states[-1][0])/5000
                daly = self.states[-1][3] * 10 + (0.81*0.051+0.14*0.133+0.05*0.655)*self.window/365 * self.states[-1][1]
                cost = (99-action)*(self.states[-1][0] + self.states[-1][2])*14.2
                daly_averted = daly_no_int[d_index] - daly
                #societal_cost = cost+daly*100000
                societal_cost = daly_averted*100000 - cost
                reward = societal_cost/self.N
            else:
                numcases=np.cumsum(np.diff([i[1] for i in self.states]).clip(0))
                numdeaths=np.cumsum(np.diff([i[3] for i in self.states]).clip(0))
                # reward = (numcases[-self.window] - numcases[-1]) - (99-action)*(self.states[-1][0])/5000
                daly = (numdeaths[-1]-numdeaths[-self.window]) * 10 + (0.81*0.051+0.14*0.133+0.05*0.655)*self.window/365 * (numcases[-1]-numcases[-self.window])
                cost = (99-action)*(self.states[-1][0] + self.states[-1][2])*14.2
                daly_averted = daly_no_int[d_index] - daly
                #societal_cost = cost+daly*100000
                societal_cost = daly_averted*100000 - cost
                reward = societal_cost/self.N
            self.rewards.append(reward)
                        
        if len(self.states) >= self.duration or self.states[-1][1] >= self.max_pop:
            done = True
        state = self.states[-1]

        return state, reward, done, {}
            

class SelectObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, index = 1):
        super().__init__(env)
        self.index = index
        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(env.observation_space.low[self.index], env.observation_space.high[self.index], shape=(1,), dtype=env.observation_space.dtype)
        elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            self.observation_space = gym.spaces.Discrete(env.observation_space.nvec[self.index])
            
    def observation(self, obs):
        return obs[self.index]

class BoxToDiscreteObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, width = 100000):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), \
            "Should only be used to wrap Continuous envs."
        self.dim = env.observation_space.shape[0]
        self.width = width
        self.range = env.observation_space.high[0]-env.observation_space.low[0]
        self.min = env.observation_space.low[0]
        self.observation_space = gym.spaces.MultiDiscrete(
        [self.range/self.width for _ in range(env.observation_space.shape[0])])

    def observation(self, obs):
        import numpy as np
        val = []
        val = np.digitize(obs, np.arange(self.min,self.min+self.range,self.width), right=True)
        return val.tolist()
