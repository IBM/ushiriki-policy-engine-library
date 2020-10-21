import gym
import requests
import pandas as pd

class CovidChallengeCalibrationEnv(gym.Env):
    def __init__(self, baseuri="https://reward-service.eu-gb.mybluemix.net", userID="61122946-1832-11ea-ssss-github", base_data="calibration_input.json", driver_data="casesdata.csv", numdays = 14, duration = 180, maxpop = 10000000.0, low=[0,0,0], high=[1,1,1], token = None):
        self.uri = baseuri+"/evaluate/policy/"
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
        self.parms["Deltap"]= [casedata_['100-wits_si'].tolist() for _ in self.parms.index]
        assert len(self.parms["Deltap"][0]) == len(self.output0), "Output0 length does not match the length of the model driver"
        assert len(self.parms["Deltap"][0]) == len(self.output1), "Output1 length does not match the length of the model driver"
        assert len(self.parms["Deltap"][0]) == len(self.output2), "Output2 length does not match the length of the model driver"
        self.I0 = casedata_['active_cases'].tolist()[0]
        self.R0 = self.output2[0]
        self.D0 = self.output1[0]
        self.S0 = self.parms["susceptible"] if self.parms["susceptible"] is not None else self.N - self.I0 - self.R0 - self.D0
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
