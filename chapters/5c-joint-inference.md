---
layout: chapter
title: Joint inference of biases, beliefs, and preferences
description: Assuming the agent performs optimally can lead to mistakes in inference. Show that we can do joint inference over large space of agents. 

---

### Introduction
In the opening [chapter](/chapters/5-biases-intro) of this section, we argued that human behavior in sequential decision problems won't always conform to optimal solving of (PO)MDPs. So if our goal is learning about human beliefs and preferences from their actions (i.e. Inverse Reinforcement Learning), then we might do better with more realistic generative models for human behavior. This chapter explores how adding time inconsistency and myopic planning to agent models affects inference of preferences.

If human behavior in some decision problem always conforms exactly to a particular sub-optimal planning model, then it would be surprising if using the true generative model for inference did not help with accurate recovery of preferences. Biases will only affect some of the humans some of the time. In a narrow domain, experts can learn to avoid biases and they can use specialized approximation algorithms that achieve near-optimal performance in the domain. So our approach is to do *joint inference* over preferences, beliefs and biases and cognitive bounds. If the agent's behavior is consistent with optimal (PO)MDP solving, we will infer this fact and infer preferences accordingly. On the other hand, if there's evidence of biases, this will alter inferences about preferences. We test our approach by comparing to a model that has a fixed assumption of optimality. We show that in simple, intuitive decision problems, assuming optimality leads to mistaken inferences about preferences.

As we discussed in Chapter IV, the identifiability of preferences is a ubiquitous issue in IRL. Our approach, which does inference over a broader space of agents (with different combinations of biases), makes identification from a particular decision problem less likely in general. Yet the lack of identifiability of preferences is not something that undermines our approach. For some decision problems, the best an inference system can do is rule out preferences that are inconsistent with the behavior and accurately maintain posterior uncertainty over those that are consistent. Some of the examples below provide behavior that is ambiguous about preferences in this way. Yet we also show simple examples in which biases and bounds *can* be identified. 


### Formalization of Joint Inference
We formalize joint inference over beliefs, preferences and biases by extending the approach developing in Chapter IV. In Equation (2) of that chapter, an agent was characterized by parameters $$  \left\langle U, \alpha, b_0 \right\rangle$$. To include the possibility of time-inconsistent and Myopic agents, an agent $$\theta$$ is now characterized by a tuple of parameters as follows:

$$
\theta = \left\langle U, \alpha, b_0, k, \nu, C \right\rangle
$$

where:

- $$U$$ is the utilty function

- $$\alpha$$ is the softmax noise parameter

- $$b_0$$ is the agent's belief (or prior) over the initial state


- $$k$$ >= $$0$$ is the constant for hyperbolic discounting function $$1/(1+kd)$$

- $$\nu$$ is an indicator for Naive or Sophisticated hyperbolic discounting

- $$C \in [1,\infty]$$ is the integer cutoff point for Myopic Exploration. 

As in Equation (2), we condition on state-action-observation triples:

$$
P(\theta | (s,o,a)_{0:n}) \propto P( (s,o,a)_{0:n} | \theta)P(\theta)
$$

We obtain a factorized form in exactly the same way as in Equation (2), i.e. we generate the sequence $$\{b_i\}_{0:n}$$ of agent beliefs:

$$
P(\theta | (s,o,a)_{0:n}) \propto 
P(\theta) \prod_{i=0}^n P( a_i | s_i, b_i, U, \alpha, k, \nu, C )
$$

The likelihood term on the RHS of this equation is simply the softmax probability that the agent with given parameters chooses $$a_i$$ in state $$s_i$$. This equation for inference does not make use of the *delay* indices used by time-inconsistent and Myopic agents. This is because the delays figure only in their internal simulations. In order to compute the likelihood the agent takes an action, we don't need to keep track of delay values. 


## Examples

### Restaurant Choice: Temptation or False Beliefs?

We return to the Restaurant Choice example. As we discussed in Chapter 5a, time-inconsistent agents can produce trajectories on the MDP (full knowledge) version of this scenario that never occur for an optimal agent without noise. In our first inference example, we do joint inference over preferences, softmax noise and the discounting behavior of the agent. (We assume for this example that the agent has full knowledge and is not Myopic). We compare the preference inferences to the earlier inference approach that assumes optimality. 





### Naive/Soph/Neutral examples for Restaurant Choice Gridworld

- Recover correct inference about preferences and agent type in three simple cases (no beliefs)

- Two or three scenario example. Softmaxing is ruled out in favor of naive/soph

- Joint inference for Naive and Soph examples:

In Naive, we consider false belief about Donut South being closed, preference for Donut North over south (unlikely in the prior) and the discounting explanation. Show multimodal inference. Discuss identification issues and Bayes Occam.

For Soph, we compare false belief the Noodle is open, a positive timeCost (which is has a low prior probability), and the Soph explanation. We could mention the experiment showing that when belief / preference explanations were possible, people tended to prefer them over HD explanations (though they did generate HD explanations spontaneously).

- Big inference example (maybe in later chapter): use HMC and do inference oiver cts params. 

### Procrastination Example
HD causes big deviation in behavior. This is like smoker who smokes every day but wishes to quit.  Can you how inference gets stronger with passing days (online inference).


### Bandits

- Heavy discount rate (which looks like myopia in the NIPS paper) and is confused with an agent who has high prior that exploring is not worth it.

- Could try synthetic example of IRL bandits. Lots of arms to drive up difference between Myopic agent and optimal. Sample behavior of 1-step Myopic agent (which is presumably somewhat less exploratory). Then compare model that doesn't include it to one that does.

### Restaurant in Foreign City example
Model without myopia assumes a preference for the restaurants that are close (i.e. a prior that prefers them). If we add more restaurants, or make the variance higher and timecost lower, we can accentuate this effect. We can also make the game repeated (problem of tractability). 


