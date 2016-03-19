---
layout: chapter
title: Myopic planning
description: Agents with faster but sub-optimal planning algorithms-- myopia about rewards and myopia about exploration. 

---


### PLAN
- motivation. as we saw pomdp planning is exponential in the number of arms. even smallish problems are hard to compute. one approach is to be myopia / greedy. in bandits, coordination of your actions across time isn't so important. you can the same states available to you at every timestep. so intuitively it seems that being short terms is not as bad as in cases where taking very long terms plans is important.

the simple kind of myopia is to treat the time horizon as short. thi is a time inconsistent model -- you model your future self as not caring about anything after time k, but then your future self will do so. we can implement this easily with delays. unlike the HD agent, we now (at each step) only simulate the next k steps, and so even with many arms this is more tractable. this results in less exploration, as we see.

more interesting myopia model is boundVOI. in a study a humans on bandits, this model was a reasonable fit. the model is again time inconsistent. the agent models his future self as not updating after a delay of d (but continuing to choose based on a now fixed set of beliefs). in fact, his future self will always update after observations. one can think of the boundVOI agent as assuming a fixed cutoff for exploration after which he will only exploit. boundVOI is used in generalized bandit problems such as Bayesian Optimization of real-valued functions. even with a VOI bound / look-ahead of 1, performance can be good.

point about not accurately modeling your future exploration behavior. q-learning doesn't model it at all. sarsa models itself taking random actions but still doesn't do any information theoretic calculation to make these maximally informative --- this is done off-line. 

One step look ahead stochastic bandits. Optimal for 2-arm bandits? For speed up, might need to rewrite the fast belief update. Have Daniel look at this. 

Monte Carlo example? Use rejection sampling instead of enumerate to compute the expected utility. ? Was in the MDP setting. Is it worth doing?  

Can we do it for stochastic bandits? You have some arm that has really low probability of being bad. So you normally wouldn't try it but you pull it if you sample (with non-trivial probability). Bandits with dangerous arms.  

### Introduction
In the previous chapter, we extended our earlier agent model for solving MDPs optimally to a model of planning for hyperbolic discounters. The goal was to better capture human behavior by incorporating one of the most prominent and well studied human *biases*. As we discussed [earlier](/chapters/5-biases-intro), any bounded agents will be unable to solve certain computational problems optimally. So when modeling human behavior (e.g. for Inverse Reinforcement Learning), we might produce better generative models by incorporating planning algorithms that are sub-optimal but which perform well given human computational bounds (i.e. they are "resource rational" in the sense of CITE).

In Chapter [POMDPs], we noted that solving a finite-horizon Multi-arm bandit problem is intractable in the number of arms and trials. So bounded agents will use some sub-optimal but tractable algorithm for this problem. In this chapter we describe and implement a widely-studied approach to Bandits (and POMDPs generally) that is sub-optimal but which can be very effective in practice. The approach is "myopic" or "greedy" with respect to exploration. The idea is that the agent at time $$t_0$$ assumes they can only *explore* (i.e. update beliefs from observations) up to some cutoff point $$C$$ steps into the future. After that point they just *exploit* (i.e. they gain rewards but don't update from the rewards they observe). In fact, the agent will continue to update beliefs after time $$t_0+C$$ -- like the Naive hyperbolic discounter the myopic agent has an incorrect model of its future self. This myopic approach to Bandits is not only useful for solving Bandit problems efficiently, but also provides a good fit to human performance in Bandit problems. In what follows, we describe Myopic Exploration in more detail, explain how to add it to our POMDP agent model (with a few lines of extra code), and then exhibit its performance on bandit problems. 

### Myopic Exploration
