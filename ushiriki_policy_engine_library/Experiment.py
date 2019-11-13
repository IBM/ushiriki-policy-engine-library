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
from multiprocessing import Pool, current_process
import random
import json
import requests
import numpy as np
import time

class Experiment():
    """
        This is the class which defines the experiment object, and enables jobs to be added and queried.
        
        This object is designed for V1.
        
        Attributes:
            _resolution (string): The population size.
            _timeout (int): The time interval in seconds that any job can be polled.
            _realworkercount (int): The number of jobs which can concurrently be active.
            policyDimension (int): The number of elements in the policy.
            _baseuri (string): The http endpoint for the task clerk.
            _locationId (string): The unique location id.
            _userId (string): The unique user id.
            _experimentCount (int): The experiment budget.
            experimentsRemaining (int): The number of jobs that are remaining in the experiment budget.
            status (boolean): Are all posted jobs known to be completed.

        """
    def __init__(self, baseuri, apiKey, userId, experimentCount = 100, actionRangeList=[], locationId = "abcd123", resolution = "test", timeout = 0, realworkercount = 1, experimentId = None, rewardType = 'cpda'):
        """
        The constructor for experiment class. If the class is being intialized with an existing experiment
        
        Parameters:
        baseuri (string): The http endpoint for the task clerk.
        apiKey (string): A valid token from the identity services for the given userId. Required
        userId (string): A valid userId able to create experiments and post jobs for a given locationId.
        experimentCount (int): The experiment budget (the number of jobs permitted) for this experiment.
        actionRangeList (list): The list of potential actions and the ranges which are in scope for jobs in this experiment.
        locationId (string): The locationId for all jobs in this experiment. Must be valid, and should have at least one reward function.
        resolution (string): The population size for the selected location. May not be valid for all locations.
        timeout (int): The time interval in seconds that any job can be polled.
        realworkercount  (int): The number of jobs which can concurrently be active.
        experimentId (string): The id of the experiment to be represented by this class.

        Returns:
        None
            
        Raises:
        ValueError
            If post operation fails for whatever reason

        """
            
        setupExperiment = '/api/v1/experiments/setupExperiment'
        
        self._resolution = resolution
        self._timeout = timeout
        self._realworkercount = realworkercount

        self.policyDimension = 2
        self._baseuri =  baseuri
        self._locationId = locationId
        self._userId = userId
        self._experimentCount = experimentCount
        self.experimentsRemaining = experimentCount
        self.status = True
        self._timestamp = time.time()
        self._apiKey = apiKey
        self.rewardType = rewardType
        
        if experimentId == None:
            data = dict([])
            data["actionRangeList"]=actionRangeList
            data["algorithmId"] = "string"
            data["exp_type"] = "string"
            data["locationId"] = self._locationId
            data["resolution"] = self._resolution
            data["resourceExperiment"] = "string"
            data["status"] = self.status
            data["timestamp"] = self._timestamp
            data["userId"] = self._userId

            response = requests.post(self._baseuri+setupExperiment, data = json.dumps(data), headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'token':self._apiKey});
            
            if response.status_code is not 200: raise ValueError("Not a valid post request")
            responseData = response.json();
            self.experimentId = responseData['jsonNode']['response']['id']
        else:
            self.experimentId = experimentId

    def getJobReward(self, jobId):
        """
            The function to poll the rewards for a given job, assuming that the job is complete.
            
            Parameters:
            jobId (string): The ID of the job that is to be queried.
            
            Returns:
            value: A dictionary containing all of the rewards associated with the provided jobId or None if the job is not complete.
            
            """
        
        getJobRewardUrl='/api/v1/experiments/reward/%s'
        if self.getJobStatus(jobId) is False:
            return None
        try:
            response = requests.post(self._baseuri+getJobRewardUrl%jobId, headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'token':self._apiKey});
            responseData = response.json()
            datatype = type(responseData['jsonNode']['rewardsObject']['ocp'][sorted(responseData['jsonNode']['rewardsObject']['ocp'].keys())[-1]])
            if datatype == list:
                value = float(responseData['jsonNode']['rewardsObject'][self.rewardType][sorted(responseData['jsonNode']['rewardsObject'][self.rewardType].keys())[-1]][0])
            elif datatype is float:
                value = float(responseData['jsonNode']['rewardsObject'][self.rewardType][sorted(responseData['jsonNode']['rewardsObject'][self.rewardType].keys())[-1]])
            else:
                value = None
        except Exception as e:
            print(e);
            value = None
        return value

    def getJobRewardBlocking(self, jobId, pollingInterval = 0, count = 1):
        """
            The function to poll the rewards for a given job, assuming that the job is complete.
            
            Parameters:
            jobId (string): The ID of the job that is to be queried.
            pollingInterval (int): The time interval in seconds that any job can be polled. Randomized.
            count (int): The number of times that a given job will be queried.

            Returns:
            reward: A dictionary containing all of the rewards associated with the provided jobId or None if the job is not complete.
            
            """
        reward = self.getJobReward(jobId)
        while reward is None and count > 0:
            time.sleep(pollingInterval+random.randint(0,6));
            count -= 1
            reward = self.getJobReward(jobId)
        if count == 0: reward = float('NaN')
        return reward

    def getJobStatus(self, jobId):
        """
            The function to poll the completion status for a given job.
            
            Parameters:
            jobId (string): The ID of the job that is to be queried. The ID must be a valid.

            Raises:
            ValueError
                If there is an attempt to check the status where JobID is null

            Returns:
            status: True iff the jobId is valid and the job has completed running.
            
            """
        status = False
        if jobId is None: raise ValueError("None as jobID")
        try:
            getJobStatusUrl='/api/v1/experiments/status/%s'
            response = requests.get(self._baseuri+getJobStatusUrl%jobId, headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'token':self._apiKey});
            status = response.text == 'true'
        except Exception as e:
            print(e);
        return status

    def getOutput(self, jobId, type):
        """
            The function to poll the completion status for a given job.
            
            Parameters:
            jobId (string): The ID of the job that is to be queried. The ID must be a valid.
            type (string): The requested type of outputfile associated with the job.

            Returns:
            status: The text for the outputfile.
            
            """
        getJobOutputUrl = '/api/v1/experiments/output/%s/%s'
        if self.getJobStatus(jobId) is False:
            return None
        response = requests.get(self._baseuri+getJobOutputUrl%(jobId, type), headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'token':self._apiKey});
        response = requests.post(rewardUrl, headers = {'Content-Type': 'application/json', 'Accept': 'application/json'})
        output = response.text
        return output

    def _postJob(self, job, seed = None):
        """
            The non-blocking function to post a job to the task clerk.
            
            Parameters:
            job (object): The job to be posted.
            seed (int): The random seed to be applied to the job.
            
            Returns:
            status: The jobId generated for this job.
            
            """
        postJobUrl='/api/v1/experiments/postJob'
        jobId = None
        intervention_names=['ITN', 'IRS', 'GVI']

        if seed is None:
            seed = random.randint(0,100)

        try:
            interventionlist = []
            for intervention in job:
                interventionlist.append( {"modelName":intervention_names[int(intervention[0])],"coverage":intervention[2], "time":"%s"%int(intervention[3])} )
            data = json.dumps({"actions":interventionlist, "experimentId": self.experimentId, "actionSeed": seed});

            response = requests.post(self._baseuri+postJobUrl, data = data, headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'token':self._apiKey});
            responseData = response.json();

            if responseData['statusCode'] == 202:
                jobId = responseData['jsonNode']['response']['id']
            else:
                message = responseData['message']
                #print(message)
                if "has been run before" in message:
                    jobId = message.split()[17]
                else:
                    raise RuntimeError(message)
        except Exception as e:
            print(e);
        return jobId

    def _postBulkJobs(self, jobs, seeds = None):
        """
            The non-blocking function to post a job to the task clerk.
            
            Parameters:
            job (object): The one or more job to be posted.
            seed (int): The random seed to be applied to the job(s).
            
            Returns:
            status: The jobId generated for this job.
            
            """
        postJobUrl='/api/v1/experiments/postBulkJobs'
        jobId = None
        intervention_names=['ITN', 'IRS', 'GVI']
        
        if type(jobs) is list and seeds is None:
            seeds = []
            for i in jobs:
                seeds.append(random.randint(0,100))
        elif type(jobs) is not list and type(seeds) is list:
            seeds = [seeds[0]]

        try:
            data = []
            
            for job,seed in zip(jobs, seeds):
                interventionlist = []
                for intervention in job:
                    interventionlist.append( {"modelName":intervention_names[int(intervention[0])],"coverage":intervention[2], "time":"%s"%int(intervention[3])} )
                data.append({"actions":interventionlist, "experimentId": self.experimentId, "actionSeed": seed, "locationId":self._locationId, "resolution":self._resolution, "userId":self._userId});

            response = requests.post(self._baseuri+postJobUrl, data = json.dumps(data), headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'token':self._apiKey});
            responseData = response.json();

            print(responseData['jsonNode'])

            if responseData['statusCode'] == 202:
                jobIds = responseData['jsonNode']['created']+responseData['jsonNode']['duplicate']
            else:
                raise RuntimeError(message)
        except Exception as e:
            print(e);
        return jobIds
    def _postJobBlocking(self, job, count = 500, seed = None):
        """
            The function to post a job to the task clerk, but blocks until the process is complete (or terminated).
            
            Parameters:
            job (object): The job to be posted.
            count (int): The number of times that a given job will be queried.
            seed (int): The random seed to be applied to the job.
            
            Returns:
            reward: A dictionary containing all of the rewards associated with the provided jobId or None if the job is not complete.
            
            """
        id = self._postJob(job, seed = seed)
        return self.getJobRewardBlocking(id, pollingInterval = self._timeout, count = count)

    def reset(self):
        """ Resets the experiment budget to the value defined at creation."""
        self.experimentsRemaining = self._experimentCount
    
    def evaluateReward(self, data):
        """
            The function to post a job to the task clerk, but blocks until the job is evaluated (or abandoned).
            
            Parameters:
            data (ndarray): The job or jobs to be evaluated.
            Raises:
            ValueError
                If Request would exceed the permitted number of Evaluations
                If the permitted number of evaluations have been exceeded
                
            Returns:
            reward: A list containing either the rewards associated with the provided job or None if the job is not complete for all jobs to be evaluated.
            
            """
        from numpy import ndarray
        print(self.experimentsRemaining, " Evaluations Remaining")
        if self.experimentsRemaining <= 0:
            raise ValueError('You have exceeded the permitted number of Evaluations')
#        if type(data) is not ndarray:
#            raise ValueError('argument should be a numpy array')

        from multiprocessing import Pool
        if (type(data) == list and type(data[0]) == list) or len(data.shape) == 2: #array of policies
            if type(data) is ndarray:
                self.experimentsRemaining -= data.shape[0]
            if type(data) == list:
                self.experimentsRemaining -= len(data)
            if self.experimentsRemaining < 0:
                raise ValueError('Request would exceed the permitted number of Evaluations')
            pool = Pool(self._realworkercount)
            result = pool.map(self._postJobBlocking, data)
            pool.close()
            pool.join()
        else:
            result = self._postJobBlocking([data])
            self.experimentsRemaining -= 1
        return result
    
    def evaluateRewardNonBlocking(self, data, coverage=.8, timeout=0):
        """
            The function to post a job to the task clerk, without blocking until the job is evaluated (or abandoned).
            
            Parameters:
            data (ndarray): The job or jobs to be evaluated.
            coverage (number): The fraction of the jobs which must be completed prior to termination
            timeout (number): The amount of time required before trying again
            Raises:
            ValueError
                If Request would exceed the permitted number of Evaluations
                If the permitted number of evaluations have been exceeded
                
            Returns:
            reward: A list containing either the rewards associated with the provided job or None if the job is not complete for all jobs to be evaluated.
            
            """
        from numpy import ndarray
        import timeit
        print(self.experimentsRemaining, " Evaluations Remaining")
        if self.experimentsRemaining <= 0:
            raise ValueError('You have exceeded the permitted number of Evaluations')
        if type(data) is not ndarray:
            raise ValueError('argument should be a numpy array')

        try:
            completed = []
            rewards = np.empty((data.shape[0]))
            rewards[:] = np.nan
            completed_new = np.empty((data.shape[0]))
            completed_new[:] = False
            envs = []
            idx = {}
            for item in data:
                envs.append(self._postJob(item))
                idx[envs[-1]] = len(envs)-1
            envs = np.array(envs)
            env_idx = np.arange(data.shape[0])
            #env_idx, envs = map(postActionWrapper, zip(np.arange(data.shape[0]),data))
            while completed_new.sum() < float(coverage)*data.shape[0]:
                timeit.time.sleep(timeout);
                tmp_envs = envs[completed_new == False]
                tmp_env_idx = env_idx[completed_new == False]
                for item in tmp_envs:
                    status = self.getJobStatus(item)
                    completed_new[idx[item]] = status == True
            for i,v in enumerate(completed_new):
                if v:
                   rewards[i] = self.getJobRewardBlocking(envs[i])
            val = rewards.tolist();
        except:
            print(exc_info(),data)
            val = None
        return val


    def evaluateRewardNonBlockingV2(self, data, coverage=.8, timeout=0):
        """
            The function to post a job to the task clerk, without blocking until the job is evaluated (or abandoned).
            
            Parameters:
            data (ndarray): The job or jobs to be evaluated.
            coverage (number): The fraction of the jobs which must be completed prior to termination
            timeout (number): The amount of time required before trying again
            Raises:
            ValueError
                If Request would exceed the permitted number of Evaluations
                If the permitted number of evaluations have been exceeded
                
            Returns:
            reward: A list containing either the rewards associated with the provided job or None if the job is not complete for all jobs to be evaluated.
            
            """
        from numpy import ndarray
        import timeit
        print(self.experimentsRemaining, " Evaluations Remaining")
        if self.experimentsRemaining <= 0:
            raise ValueError('You have exceeded the permitted number of Evaluations')
        if type(data) is not ndarray:
            raise ValueError('argument should be a numpy array')

        try:
            completed = []
            rewards = np.empty((data.shape[0]))
            rewards[:] = np.nan
            completed_new = np.empty((data.shape[0]))
            completed_new[:] = False
            envs = []
            idx = {}





            from multiprocessing import Pool
            if len(data.shape) >= 2: #array of policies
                if type(data) is ndarray:
                    self.experimentsRemaining -= data.shape[0]
                if self.experimentsRemaining < 0:
                    raise ValueError('Request would exceed the permitted number of Evaluations')
                pool = Pool(self._realworkercount)
                envs = pool.map(self._postJob, data)
                pool.close()
                pool.join()
            else:
                envs = [self._postJob(data)]
                self.experimentsRemaining -= 1

            idx = {env:loc for loc,env in enumerate(envs)}
            print(envs, idx)
            '''
            for item in data:
                envs.append(self._postJob(item))
                idx[envs[-1]] = len(envs)-1
            '''

            envs = np.array(envs)
            env_idx = np.arange(data.shape[0])
            #env_idx, envs = map(postActionWrapper, zip(np.arange(data.shape[0]),data))
            while completed_new.sum() < float(coverage)*data.shape[0]:
                timeit.time.sleep(timeout); 
                tmp_envs = envs[completed_new == False]
                tmp_env_idx = env_idx[completed_new == False]
                print(completed_new, tmp_env_idx)
                for item in tmp_envs:
                    status = self.getJobStatus(item)
                    completed_new[idx[item]] = status == True
                    print(item, status)
            for i,v in enumerate(completed_new):
                if v:
                   print(envs[i])
                   rewards[i] = self.getJobRewardBlocking(envs[i])
            val = rewards.tolist();
        except:
            print(exc_info(),data)
            val = None
        return val
