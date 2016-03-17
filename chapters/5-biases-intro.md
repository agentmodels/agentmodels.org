---
layout: chapter
title: "Cognitive biases and bounded rationality"
description: Discuss soft-max noise, limited memory, heuristics/biases, motivation from intractability of POMDPs.
is_section: true
---

- hawthorne paper

- discuss which biases show up for single decisions already, and which require multiple sequential decisions?


### Optimality and modeling human actions

Earlier chapters describe agent models for *optimal* decision making and planning in sequential decision problems (MDPs and POMDPs). These models can serve as a guide when we build an algorithm that makes good choices. If we can write down a utility function (which is not always easy -- see below) and define the relevant state and action spaces, then we can use the kind of agent models for MDPs and POMDPs described earlier (or approximations of those models). Creating algorithms that make good choices is a common application in AI and robotics, OR and engineering and in applied economics (e.g. algorithmic trading).

The previous chapter discussed the problem of learning the utility functions and beliefs of agents from their behavior. The MDP and POMDP agent models serve as *generative* models of behavior that can be inverted to learn about the agent. The models are a means to the end of modeling the agent's behavior. If the agent does not behave optimally, we'd be better off (all other things being equal) using a different model. 

Our main application for learning about agents is learning about human preferences. So for a given decision problem, we should ask whether human choices will be optimal (in the MDP or POMDP sense). If not, we might consider using an alternative generative model. One common alternative model has already been discussed: the agent with softmax noise. This model has been used in various IRL papers to deal with human deviations from optimality [cite: young dialogue turn-taking paper, zheng/zhang taxi paper]. The softmax model is analytically tractable and allows one to use existing MDP and POMDP solution approaches. Yet it has limited expressiveness. Softmax is a a model of *random* deviations from optimality. By averaging over multiple draws of the agent's behavior, the agent's optimal action can be directly inferred. (In other words: even without knowing $$\alpha$$ we still have an unbiased estimator of the agent's maximizing action. Likewise for other random noise models.) However humans may deviate from optimality *systematically*. That is, they may take a sub-optimal action (relative to their own preferences) with higher probability than the optimal action. [Examples of systematic errors: a student making persistent errors on a series of exams that reflect some misunderstanding. Optical illusions.] In that case, we will need to consider generative models other than the softmax model.


### Human deviations from optimal action - Cognitive Bounds
Humans deviate from optimal behavior in sequential problems in myriad ways. Our focus is not on all such deviations. We focus on cases where deviations in MDP/POMDP problems are both substantial (such that they significantly alter preferences) and systematic (so cannot be captured by the softmax model or something simiar). For practical purposes, deviations will be less important if the relevant preferences and beliefs can be learned via some other method. 

A first class of deviation is cognitive or computational *bounds* (in the sense of *bounded rationality* or *bounded optimality*). Humans (and AI agents) will perform suboptimally on some MDPs and POMDPs due to basic computational constraints. For example, consider a POMDP where the agent learns about the locations of restaurants on a grid representing a large city and then navigates between them. The ideal POMDP agent will never forget a location once it has been observed. Yet a human would not recall the exact location of hundreds of restaurants from a single walk through a city. Moreover this forgetting will produce behavior that differs systematically from the optimal agent. 

Even simple POMDPs are intractable as they scale up in size [cite]. So for any finite agent there will be POMDPs that they cannot solve exactly. It's plausible that humans often encounter POMDPs of this kind. For example, in lab experiments humans make systematic errors in computationally trivial bandit problems [yu, michael lee papers --- see also the paper by finale and zoubin on humans in tiger and other basic POMDPs]. Thus it's likely that humans will make errors on more complex POMDPs. (Consider the example of designing scientific experiments. This problem has a long time horizon (e.g. a lab could perform hundreds of experiment in a year). The inference from a single experiment is challenging and combining experimental results is hard.) In a chapter in this section [link], we implement the kind of computational bound (or bias) that has been used to model actual human performance in bandit problems. 


### Human deviations from optimal action - Cognitive Biases
Cognitive bounds will apply to any finite agent. Humans also have cognitive *biases* that may be more idiosyncratic (and less likely to apply to artifical agents). There is a large literature on cognitive biases in psychology and behavioral economics [cite reviews]. One particularly relevant example are the biases summarized by "Prospect Theory" (including Loss Aversion and Framing Effects). These are biases in decisions between simple one-shot lotteries over money or other prizes. We'd expect them to lead to systematics errors in real-life scenarios modeled as (PO)MDPs (assuming people don't use computers for all their expected value calculations). Another important bias is *time inconsistency*. This bias has been used to explain addiction, procrastination, impulsive behavior and the use of pre-commitment. We describe the bias and implement time-inconsistent planning in the next chapter. 


### Learning preferences from bounded and biased agents
We've argued that humans have biases and bounds that are systematic (and so can't be captured by the softmax noise model) and relevant to their actions in (PO)MDPs. In order to infer human beliefs and preferences when these biases are present, we can extend our (PO)MDP agent generative in order to capture these biases. The next two chapters describe and implement models for time-inconsistent and myopic (near-sighted) planning. The final chapter in this section implements inference for agents with these biases. Here we show that these biases make a significant difference in terms of inferring preferences correctly. We also show that we can do inference over a large space of agents, by combining unbiased agents with various flavors of biased agent. 



### Plan for rest of section
- Describe time inconsistency. 
- Show model for time inconsistent agent. Show gridworld examples.
- Describe myopia and bound VOI. Show bandit problems and maybe one of the gridworld examples.
- Then show inference about these agents. Multimodal inference using everything that's happened before with nice graphs!

