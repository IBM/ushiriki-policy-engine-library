
#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='ushiriki_policy_engine_library',
      version='1.1',
      description='Python Langauge User facing library for accessing the Ushiriki Policy Engine webservice API',
      author='Ushiriki Policy Engine Library Developers',
      author_email='malaria.ml.challenge@ke.ibm.com',
      url='https://github.com/IBM/ushiriki-policy-engine-library',
      install_requires = ["gym", 
          "requests", 
          "samplemetamodel @ git+https://github.com/Model-Driven-Discovery-Workshop/samplemetamodel",
          "covasimmetamodels @ git+https://github.com/Model-Driven-Discovery-Workshop/covasimmetamodels"],
      packages=find_packages(),
)
