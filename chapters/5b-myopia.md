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
In the previous chapter, we extended our earlier agent model for solving MDPs optimally to a model of planning for hyperbolic discounters. The goal was to better capture human behavior by incorporating one of the most prominent and well studied human *biases*. As we discussed [earlier](/chapters/5-biases-intro), any bounded agent will be unable to solve certain computational problems optimally. So when modeling human behavior (e.g. for Inverse Reinforcement Learning), we might produce better generative models by incorporating planning algorithms that are sub-optimal but which perform well given human computational bounds (i.e. they are "resource rational" in the sense of CITE). This chapter describes an efficient planning algorithm which sometimes performs close to optimally. The goal is to illustrate how our existing agent models can be extended to capture boundedly rational behavior by adding only a few lines of code. 

### Myopic Exploration: the basic idea
In Chapter [POMDPs], we noted that solving a finite-horizon Multi-arm bandit problem is intractable in the number of arms and trials. So bounded agents will use some sub-optimal but tractable algorithm for this problem. In this chapter we describe and implement a widely-studied approach to Bandits (and POMDPs generally) that is sub-optimal but which can be very effective in practice. We refer to the approach as *Myopic Exploration*, because it is "myopic" or "greedy" with respect to exploration. The idea is that the agent at time $$t_0$$ assumes they can only *explore* (i.e. update beliefs from observations) up to some cutoff point $$C$$ steps into the future. After that point they just *exploit* (i.e. they gain rewards but don't update from the rewards they observe). In fact, the agent will continue to update beliefs after time $$t_0+C$$; like the Naive hyperbolic discounter the myopic agent has an incorrect model of its future self. We call an agent that uses Myopic Exploration a "Myopic Agent". This will be precisely defined below. 

Myopic Exploration is not only useful for solving Bandit problems efficiently, but also provides a good fit to human performance in Bandit problems. In what follows, we describe Myopic Exploration in more detail, explain how to incorporate it into out POMDP agent model, and then exhibit its performance on Bandit problems.

### Myopic Exploration: applications and limitations
As noted above, Myopic Exploration has been studied in Machine Learning refp:gonzalez2015glasses and Operations Research refp:ryzhov2012knowledge as part of algorithms for generalized Bandit problems. In most cases, the cutoff point $$C$$ after which the agent assumes himself to exploit is set to $$C=1$$. This results in a scalable, analytically tractable optimization problem: pull the arm that maximizes the expected value of future exploitation given you pulled that arm. This "future exploitation" means that you pick the arm that is best in expectation for the rest of time. The Myopic Agent with $$C=1$$ has also been successfully used a model of human play in Bandit problems refp:zhang2013forgetful. 

We've presented Bandit problems with a finite number of arms, and with discrete rewards that are uncorrelated across arms. Myopic Exploration works well in this setting but also works for generalized Bandit Problems: e.g. when rewards are correlated, when rewards are continuous, and in the "Bayesian Optimization" setting where instead of a fixed number of arms the goal is to optimize high-dimensional real-valued function refp:ryzhov2012knowledge. 

Myopic Exploration will not work well for POMDPs in general. Suppose I'm looking for a good restaurant in a foreign city. A good strategy is to walk to a busy street and then find the restaurant with the longest line. If reaching the busy street takes longer than the myopic cutoff $$C$$, then a Myopic agent would not model himself as learning which restaurant has the longest line -- and hence would not recognize this as a good strategy. The Myopic agent would only carry out this strategy if it could observe the restaurants before the cutoff $$C$$. (This kind of POMDP could easily be represented in our Gridworld framework.). This highlights a way in which Bandit problems are distinctive from general POMDPs. In a Bandit problem, you can always explore every arm: you never need to first move to an appropriate state. So even the Myopic Agent with $$C=1$$ compares the information value of every possible observation that the POMDP can yield.

The Myopic agent has an incorrect model of his future self, assuming his future self stops updating after cutoff point $$C$$. This incorrect "self-modeling" is also a property of well-known model-free RL agents. For example, a Q-learner's estimation of expected utilities for states ignores the fact that the Q-learner will randomly explore with some probability. SARSA, on the other hand, does take its random exploration into account when computing this estimate. But it doesn't model the way in which its future exploration behavior will make certain actions useful in the present (as in the example of finding a restaurant in a foreign city).

### Myopic Exploration: formal model
Myopic Exploration only makes sense in the context of an agent that is capable of learning from observations (i.e. in the POMDP rather than MDP setting). So our goal is to generalize our agent model for solving POMDPs to a Myopic Exploration with $$C \in [1,\Infinity]$$.

**Exercise:** Before reading on, modify the equations defining the [POMDP agent](/chapters/3c-pomdp) in order to generalize the agent model to include Myopic Exploration. The optimal POMDP agent will be the special case when $$C=\Infinity$$.

------------

To extend the POMDP agent to the Myopic agent, we use the idea of *delays* from the previous chapter. These delays are not used to evaluate future rewards (as any discounting agent would use them). They are used to determine how future actions are simulated. If the future action occurs with delay $$d$$ past the cutoff point $$C$$, i.e. $$d \gte C$$, then the simulated future self does not do a belief update before taking the action. (This makes the Myopic agent analogous to the Naive agent: both simulate the future action by projecting the wrong delay value onto their future self). 

We retain the notation from the definition of the POMDP agent and skip directly to the equation for the expected utility of a state, which we modify for the Myopic agent with cutoff point $$C \in [1,\Infinity]$$:

$$
EU_{b}[s,a,d] = U(s,a) + E_{s',o,a'}(EU_{b'}[s',a'_{b'},d+1])
$$

where (as before):

- $$s' \sim T(s,a)$$ and $$o \sim O(s',a)$$

- $$a'_{b'}$$ is the softmax action the agent takes given belief $$b'$$

- If $$d<C$$ the new belief state $$b'$$ is defined (as before):

$$
b'(s') \propto O(s',a,o)\sum_{s \in S}{T(s,a,s')b(s)}
$$

- Otherwise, $$b'$$ is defined as:

$$
b'(s') \propto \sum_{s \in S}{T(s,a,s')b(s)}
$$

The key part is the definition of $$b'$$ when $$d \gte C$$. The Myopic agent assumes his future self updates only on his last action $$a$$ and not on observation $$o$$. So the future self will know about state changes that follow a priori from his actions. (In a deterministic Gridworld, the future self would know his new location and that the time remaining had been counted down).

The implementation of the Myopic agent in WebPPL is a direct translation of the definition provided above.

**Exercise:** Modify the code for the POMDP agent [todo link to codebox] to represent a Myopic agent.


### Myopic Exploration for Bandits

We show the performance of the Myopic agent on Multi-Arm bandits.

For 2-arms, Myopic with D=1 is optimal. Verify this and compare runtime.

For >2 arms, I believe Myopic D=1 is not optimal. Verify this. It should be much faster as the number of arms grows. (One easy way to speed it up is to have a special *updateBelief* in *beliefDelayAgent* for stochastic bandits. the only difference is that once delay>=C, you should just directly update the timeLeft, assuming you have belief in *ERPOverLatentState*. This will avoid the Enumerate for *nextBelief*.)















In this tutorial, we consider only Bandit problems with a finite number of arms and a fixed discrete set of rewards. 

- uses of myopic. bandit problems generalized: continuous rewards / correlated rewards, function optimization. won't work in gridworld like settings. for example, it might be good to visit a street not because i know where to go but because i'll see the restaurant with a line and go there. if i assume my future self will stop updating, i can't realize the goodness here. in bandits, the world has simple structure -- i can test any arm at any timestep and so myopia seems much less a restriction. 

- as noted, is like naive planning in false self model. in this way, also like model-free RL -- q learning and sarsa. 

- human behavior -- cite Yu.  




