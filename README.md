# ushiriki-policy-engine-library
User facing library for accessing the Ushiriki Policy Engine webservice API


To install this library:
`pip install https://github.com/IBM/ushiriki-policy-engine-library --user --upgrade`

This command will download and install the current source code from this repo (upgrading the source code if previously installed). It will also place the necessary modules in the local [user] space. All dependencies required will also be installed as well. Alternatively, one could also clone the repository, however the path parameters and the requirements will need to be manually configured.

Once installed, this will permit python programs executed by the user who performed this installation to access program commands such as: 

 ```python
from ushiriki_policy_engine_library.SimpleChallengeEnvironment import ChallengeEnvironment
from ushiriki_policy_engine_library.EvaluateSubmission import EvaluateAugmentedChallengeSubmission,  EvaluateChallengeSubmission
```

Once these modules are imported, the environments required are available for use.
The only remaining task is to provide the necessary user credentials, and to define the location where the environment is running.

An example of these configurations for the sample environment are as follows:

```python
class ChallengeEnvironment1(ChallengeEnvironment):
    def __init__(self):
        ChallengeEnvironment.__init__(self,baseuri="http://127.0.0.1:8080", userID="61122946-1832-11ea-8d71-362b9e155667")
```     

This code creates an environment object, and defines a user. Some environments will require additional parameters such as tokens and other optional elements.

## Citation

```lisp
@misc{pending,
  Author = {Sekou L Remy and Oliver E Bent},
  Title = {A Global Health Gym Environment for RL Applications},
  Year = {2020},
  Eprint = {arXiv:pending},
}
```
