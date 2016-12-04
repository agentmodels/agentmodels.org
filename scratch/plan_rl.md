### RL chapter plan

## Introduction

The previous chapter introduced POMDPs: decision problems where some features of the environment are initially unknown to the agent but can be learned by observation. We showed how to compute optimal Bayesian behavior for POMDPs. Unfortunately, this computation is infeasible for all but the simplest POMDPs. In practice, many POMDP problems can be solved heuristically using "Reinforcement Learning" (RL). RL algorithms are conceptually simple, scalable and effective both in discrete and continuous state spaces. They are central to achieving state-of-the-art performance in sequential decision problems in AI, e.g. in playing Go [cite], in playing videogames from raw pixels [cite], and in robotic control [cite]. 

## Reinforcement Learning for Bandits
The previous chapter showed how the optimal POMDP agent solves Bandit problems. Here we apply Reinforcement Learning to Bandits.

### Softmax Greedy Agent
First, we consider a "greedy" agent with softmax noise. This agent updates beliefs about the hidden state in the same way as the POMDP agent. Yet instead of making sequential plans that balance exploration (e.g. making informative observations) with exploitation (gaining high reward), the Greedy agent simply takes the action with highest immediate expected return[^greedy]. Here we implement the Greedy agent on Bernoulli Bandits. 

[^greedy]:In a later chapter, we implement a more general Greedy/Myopic agent by extending the POMDP agent. Here we implement the Greedy agent from scratch and apply it to Bernoulli Bandits.

CODEBOX: graph showing expected regret (reward vs. expected reward of optimal agent). print running time.

>*Exercise*:

> 1. Set the softmax noise to be low. How well does the Greedy Softmax agent do? Explain why. Keeping the noise low, modify the agent's priors to be overly "optimistic" about the expected reward of each arm (without changing the support of the prior distribution). How does this optimism change the agent's performance? Explain why. (This idea is known as "optimism in the face of uncertainty" in the RL literature.)

> 2. Modify the agent so that the softmax noise is low and the agent has a "bad" prior (i.e. one that assigns a low probability to the truth) that is not optimistic. Will the agent eventually learn the optimal policy? How many trials does it take on average?

How does this Greedy agent compare to the optimal POMDP agent? As the graph shows, the Greedy agent quickly converged on the same policy the optimal agent would learn. More generally, Greedy agents perform well empirically (Precup) and with slight modifications achieve optimal asymptotic performance (Cesa-Bianchi). Moreover, the Greedy agent runs scales well in the number of arms and the horizon, while the optimal agent soon becomes intractable. 

The optimal POMDP agent solves a harder problem. It computes what to do for any possible sequence of observations. This means the POMDP agent, after computing a policy once, could immediately take the optimal action given any sequence of observations without doing any more computation. By contrast, RL agents store information only about the present Bandit problem -- and in most Bandit problems this is all we care about. 

CODEBOX: could compare the Greedy agent and optimal agent on a small Bandit problem. do a few runs and compare their expected regret.



### Posterior sampling
Posterior sampling (or "Thompson sampling") is another algorithm for Bandits and for general RL problems. The Posterior-sampling agent updates beliefs like the POMDP agent. Before choosing an arm, it draws a sample from its posterior on the arm parameters and then chooses greedily given the sample. In Bandits, this is similar to Softmax Greedy but without the temperature parameter $$\alpha$$.

CODEBOX: same graphs as above. Could show Thompson sampling for Gaussian and correlated Gaussian arms. 


## Reinforcement Learning for MDPs

### POMDP vs. RL
The previous chapter introduced a POMDP version of the Restaurant problem in Gridworld, where the agent doesn't know initially if each restaurant is open or closed. How would RL agents compare to POMDP agents on this problem?

One way to think of the Restaurant Choice problem is as an *Episodic* POMDP. At the start of each episode, the agent is uncertain about which restaurants are open or closed. Over repeated episodes, they learn about the *distribution* on restaurants being open but they never know for sure (since restaurants might close down or vary their hours) and so they may need to update their beliefs on any given episode. (A similar examples in the POMDP literature is "Tiger".) This kind of problem is ill-suited to standard RL algorithms. Such algorithms assume that the hidden state is an MDP that is fixed across all episodes. POMDP algorithms, on the other hand, take into account the fact that there is new (but observable) hidden state every episode.

<!-- The general learning problem: there is some state that's initially unknown and fixed across episodes and some state that's random across episodes but observable. A POMDP agent should be able to learn both of these -->

Alternatively, we could think of the Restaurant Choice problem as an episodic MDP. Initially, the agent doesn't know which restaurants are open. But once they find out there is nothing more to learn: the same restaurants are open each episode. In this kind of example, RL techniques work well and are typically what's used in practice. 

### RL algorithms for MDPs
We now consider RL algorithms for learning arbitrary fixed MDPs. The goal is typically to learn a policy that achieves high reward as quickly as possible. Algorithms are either *model-based* or *model-free*:

1. *Model-based* algorithms learn an explicit representation of the MDP's transition and reward functions. These representations are used to compute a good policy. 

2. *Model-free* algorithms do not explicitly represent the transition and reward functions. Instead they explicitly represent either a value function (e.g. a Q- or V-function) or a policy. 

### Q-learning (TD-learning)
Q-learning is the best known RL algorithm and is model-free. A Q-learning agent stores and updates a point estimate of the expected utility of each action under the optimal policy (i.e. an estimate Q^(s,a) for Q_star_(s,a)). Provided the agent takes random exploratory actions, these estimates converge in the limit (cite Watkins). In our framework, it's more natural to implement *Bayesian Q-learning* (Dearden et al), where the point estimates are replaced with Bayesian posteriors.

The defining property of Q-learning (vs. SARSA or Monte-Carlo) is how it updates its Q-value estimates. After each state transition (s,a,r,s'), a new Q-value estimate is computed:
Q^(s,a) = r + max_a' Q(s',a')

CODEBOX: Bayesian Q-learning. Apply to gridworld where goal is to get otherside of the and maybe there are some obstacles. For small enough gridworld, POMDP agent will be quicker.

Note that Q-learning works for continuous state spaces. 

### Policy Gradient
- Directly represent the policy. Stochastic function from states to actions. (Can put prior over that the params of stochastic function. Then do variational inference (optimization) to find params that maximize score.)

### Posterior Sampling Reinforcement Learning (PSRL)

Posterior Sampling Reinforcemet Learning (PSRL) is a model-based algorithm that generalizes posterior-sampling for Bandits to discrete, finite-horizon MDPs (cite Strens). The agent is initialized with a Bayesian prior distribution on the reward function $$R$$ and transition function $$T$$ and for every episode proceeds as follows:

> 1. Sample $$R$$ and $$T$$ (a "model") from the distribution. Compute the optimal policy for this model and follow that policy until the episode ends (while storing all experiences during the episode). 

> 2. Update the distribution on $$R$$ and $$T$$ on the experiences during the episode using Bayes Rule. 

Intuition for PSRL: if very confident, agent mainly exploit a model. if unconfident then will act as if different models are true. if one plausible model says that certain states have high reward when they in fact don't, agent will sample that model and visit those states and discover that they suck. after this, the agent will update and won't consider those models again. 

Implementation. Start by defining a distribution on R only. Det version: each in gridworld either has zero/one reward. Aim to find the state with reward one. We then run agent for many episodes. (Think about display for this). Compare Q-learning on this problem. 

Gridworld maze: Agent is in a maze in perfect darkness. Each square could be wall or not with even probability. Agent has to learn how to escape. Maze could be fairly big but want a fairly short way out. Model for T. 

Clumpy reward model. Gridworld with hot and cold regions that clump. Agent starts in a random location. If you assume clumpiness, then agent will go first to unvisited states in good clumps. Otherwise, when they start in new places they'll explore fairly randomly. Could we make a realistic example like this? (Once you find some bad spots in one region. You don't explore anywhere near there for a long time. That might be interesting to look at. Could have some really cold regions near the agent.

Simple version: agent starts in the middle. Has enough time to go to a bunch of different regions. Regions are clumped in terms of reward. Could think of this a city, cells with reward are food places. There are tourist areas with lots of bad food, foodie areas with good food, and some places with not much food. Agent without clumping tries some bad regions first and keeps going back to try all the places in those regions. Agent with clumping tries them once and then avoids. 


### RL and Inferring Preferences

Most IRL is actually inverse planning in an MDP. Assumption is that it's an MDP and human already knows R and T. Paper on IRL for POMDPs: assume agent knows POMDP structure. Much harder inference problem. 

We have discussion of biases that humans have: hyperbolic discounting, bounded planning. These are relevant even if human knows structure of world and is just trying to plan. But often humans don't know structure of world. Better to think of world as RL problem where MDP or POMDP also is being learned. Problem is that there are many RL algorithms, they generally involve lots of randomness or arbitrary parameters. So hard to make precise predictions. Need to coarsen. Show example of this with Thompson sampling for Bandits. 

Could discuss interactive RL. Multi-agent case. It's beyond scope of modeling.






