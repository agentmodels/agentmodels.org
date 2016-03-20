---
layout: chapter
title: Joint inference of biases, beliefs, and utilities
description: Assuming the agent performs optimally can lead to mistakes in inference. Show that we can do joint inference over large space of agents. 

---

### PLAN

- we motivated modeling biases and bounds by saying we want better generative models for IRL. we now look at how adding biases/bounds to the model changes inference. 

- ideal: show that with big model we (1) can still make good predictions when human performs optimally. (2) we infer better when agent is not optimal. (3) we are fairly efficient.


##Examples

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


