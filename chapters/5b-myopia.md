---
layout: chapter
title: Myopic Exploration for Bounded Agents
description: Agents with faster but sub-optimal planning algorithms-- myopia about rewards and myopia about exploration. 

---


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
Myopic Exploration only makes sense in the context of an agent that is capable of learning from observations (i.e. in the POMDP rather than MDP setting). So our goal is to generalize our agent model for solving POMDPs to a Myopic Exploration with $$C \in [1,\infty]$$.

**Exercise:** Before reading on, modify the equations defining the [POMDP agent](/chapters/3c-pomdp) in order to generalize the agent model to include Myopic Exploration. The optimal POMDP agent will be the special case when $$C=\infty$$.

------------

To extend the POMDP agent to the Myopic agent, we use the idea of *delays* from the previous chapter. These delays are not used to evaluate future rewards (as any discounting agent would use them). They are used to determine how future actions are simulated. If the future action occurs when delay $$d$$ exceeds cutoff point $$C$$, then the simulated future self does not do a belief update before taking the action. (This makes the Myopic agent analogous to the Naive agent: both simulate the future action by projecting the wrong delay value onto their future self). 

We retain the notation from the definition of the POMDP agent and skip directly to the equation for the expected utility of a state, which we modify for the Myopic agent with cutoff point $$C \in [1,\infty]$$:

$$
EU_{b}[s,a,d] = U(s,a) + E_{s',o,a'}(EU_{b'}[s',a'_{b'},d+1])
$$

where:

- $$s' \sim T(s,a)$$ and $$o \sim O(s',a)$$

- $$a'_{b'}$$ is the softmax action the agent takes given new belief $$b'$$

- the new belief state $$b'$$ is defined as:

$$
b'(s') \propto I_C(s',a,o,d)\sum_{s \in S}{T(s,a,s')b(s)}
$$

$$
I_C(s',a,o,d) = O(s',a,o)$$ \mbox{if } d<C and $$=1$$ (unity) otherwise.

<!--
If $$d<C$$ the new belief state $$b'$$ is defined (as previously):

$$
b'(s') \propto O(s',a,o)\sum_{s \in S}{T(s,a,s')b(s)}
$$

On the other hand, if $$d \gte C$$, then:

$$
b'(s') \propto \sum_{s \in S}{T(s,a,s')b(s)}
$$
-->

The key part is the definition of $$b'$$ when $$d \gte C$$. The Myopic agent assumes his future self updates only on his last action $$a$$ and not on observation $$o$$. So the future self will know about state changes that follow a priori from his actions. (In a deterministic Gridworld, the future self would know his new location and that the time remaining had been counted down).

The implementation of the Myopic agent in WebPPL is a direct translation of the definition provided above.

**Exercise:** Modify the code for the POMDP agent [todo link to codebox] to represent a Myopic agent.


### Myopic Exploration for Bandits and Gridworld

We show the performance of the Myopic agent on Multi-Arm bandits.

For 2-arms, Myopic with D=1 is optimal. Verify this and compare runtime.

For >2 arms, I believe Myopic D=1 is not optimal. Verify this. It should be much faster as the number of arms grows. (One easy way to speed it up is to have a special *updateBelief* in *beliefDelayAgent* for stochastic bandits. the only difference is that once delay>=C, you should just directly update the timeLeft, assuming you have belief in *ERPOverLatentState*. This will avoid the Enumerate for *nextBelief*.)

TODO
We make a Gridworld version of the "Restaurant Search" problem. The agent is uncertain of the quality of all of the restaurants and has an independent uniform prior on each one, in particular `uniformDraw( _.range(1,11) )'. By moving adjacent to a restaurant, the agent observes the quality (e.g. by seeing how full the restaurant is or how good it looks from the menu). An image of the grid, which includes the true latent restaurant utilities and disiderata for where the agent should end up is in: /assets/img/5b-myopia-gridworld.pdf.

![myopia gridworld](/assets/img/5b-myopia-gridworld.pdf)

[TODO images cant be pdf?]

Assuming we want to stick with "no uncertainty over utilities" and "utilities depend only on state", we would have to implement this by having extra states associated with the utility values in range(1,11). The latent state is the table {restaurantA:utilityRestaurantA}. The transition function is the normal gridworld transition, with an extra condition s.t. when the agent goes to a restaurant they get sent to state corresponding to the restaurant's utility. (Whatever solution is used need not be general. We don't need to show the code, we just need to make the example work).







