[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Udacity-Collaboration-and-Competition-Project

The Jupyter-Notebook-File [Tennis.ipynb](Tennis.ipynb) includes my solution to the Collaboration-and-Competition-Project in the Deep-Reinforcement-Learning-Nanodegree.

### Introduction

In this project, two agents are trained to play tennis in the Unity ML-Agents Tennis environment. The training is done by using a variant of the DDPG algorithm for two agents. Note however that the algorithm is not precisely identical to the multi-agent DDPG from the paper ["Multi-Agent Actor Critic for Mixed Cooperative-Competitive Environments"](https://arxiv.org/pdf/1706.02275.pdf).

![Trained Agent][image1]

### Description of the Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Installation

If you want to use the [Tennis.ipynb](Tennis.ipynb)-notebook to train the two agents on your own you first have to download several packages and the environment.

To install all the necessary packages and dependencies you can set up a new python environment as explained in the README.md in the [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).
Depending on your operating system you can download the environment using one of the following links (the extraction of the file that I downloaded is the Tennis_Linux folder that is included in repository):

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

After downloading the file you have to unzip it. Then follow the instructions in the first part of the Jupyter Notebook to instantiate the environment. 

### Instructions on [Tennis.ipynb](Tennis.ipynb)-notebook

After you instantiated the environment in the first part you will examine the state and action space in the second. In the third part you can check how random agents perform in the environment.

By running the cells in the fourth section you will train your own agents on the environment and test them afterwards.  
Once the necessary classes (Actor, Critic, Agent, OUNoise and ReplayBuffer) are defined you will create two instances of the Agent-class. Then (by defining and running the ddpg()-function) the variantof the deep deterministic policy gradient (DDPG-) algorithm for two agents is carried out on those instances. 

The agents stop learning after the goal is reached (average total reward of +0.5 over 100 consecutive episodes). You can plot the performance of your agents that was achieved during training.

At the end of the notebook you can test your trained agents on 100 further episodes and check their performance by looking at the score-plot and the average score that was achieved during these 100 test episodes.
