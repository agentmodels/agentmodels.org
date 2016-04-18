---
layout: chapter
title: "Cognitive biases and bounded rationality"
description: Discuss soft-max noise, limited memory, heuristics/biases, motivation from intractability of POMDPs.
is_section: true
---


### Optimality and modeling human actions

We've mentioned two uses for models of sequential decision making:

(1). **Solve practical decision problems** (preferably with a fast algorithm that performs optimally)

(2). **Learn the preferences and beliefs of humans** (e.g. to predict future behavior or to provide useful recommendations)

The first chapters of the book focused on (1). We presented agent models for solving MDPs and POMDPs optimally. We demonstrated these models on toy problems (Gridworld and simple variants of Bandits) that are closely related to practical real-world problems. The previous [chapter](/chapters/4-reasoning-about-agents), by contrast, focused on (2). This chapter used the MDP and POMDP agent models not to solve problems but as *generative models* of human behavior. But are the MDP and POMDP agent models good models of human behavior? These are models of *optimal* action for a particular decision problem. If humans deviate from optimality for such a decision problem, will these be bad generative models?

### Random vs. Systematic Errors
The agent models we've considered are models of *optimal* performance on (PO)MDPs. Nevertheless, they are flexible models. The agent can have any utility function and any initial belief distribution. We saw in the previous chapters that apparently irrational behavior can sometimes be explained in terms of inaccurate prior beliefs.

Some kinds of human behavior resist explanation in these terms. Consider a cigarette smoker who smokes every day while wanting to quit. Such people have often quit previously and relapsed multiple times. They can be well informed both about the health effects of smoking and the psychological effects of avoiding cigarettes. It's hard to explain their behavior in terms of inaccurate beliefs[^beliefs].

[^beliefs]: One could argue that the smoker has a temporary belief that smoking is high utility which causes them to smoke. This belief subsides after smoking a cigarette and is replaced with regret. To explain this in terms of a POMDP agent, there has to be an observation that triggers this change of belief via Bayesian belief-updating. But what is this observation? The smoker may have *cravings*. Yet cravings alter the smoker's desires or wants, rather than being observational evidence about the empirical world. 

One standard response is to model deviations from optimal behavior using softmax noise. As we've seen above, it's easy from a computational standpoint to move between perfect maximizing behavior and soft-max behavior. Various papers doing Inverse Reinforcement Learning (IRL) on human data take this approach refp:kim2014inverse and refp:zheng2014robust. Yet the softmax model also has limited expressiveness. It's a model of *random* deviations from optimal behavior. Bigger deviations from optimal are explained by more randomness overall. Models of random error might be a good fit for certain motor or perceptual tasks (e.g. throwing a ball or locating the source of a distant sound). But the smoking example suggests that humans deviate from optimality *systematically*. That is, when not behaving optimally, their actions may still be predictable and bigger deviations from optimality do not imply more randomness.

Here are some examples of systematic deviations from optimal action:
<br>
>**Systematic deviations from optimal action**

- Smoking every week (i.e. systematically) while simultaneously trying to quit (e.g. using patches, throwing out cigarette packets).

- Always completing assignments just before the deadline, while always planning to complete the assignment as early as possible. 

- Forgetting random strings (passwords, ID numbers, words in a foreign language) within a few hours -- assuming they weren't explicitly memorized[^strings].

- Failing frequently on the same kind of math problem (e.g. long integer division).

[^strings]: With effort people can memorize these strings and store them in mind for longer periods. The claim is that if people do not make an attempt to memorize a random string, they will systematically forget the string within a short duration. This can't be easily explained on a POMDP model, where the agent has perfect memory.

This is not to say that all human deviations are systematic. 



| Goal | Key tasks | Find optimal solution? | Subject Areas | Fields |
|:--------|:-------:|:--------:|--------:|
| Build optimal decision making systems |
1. Define appropriate utility function and decision problem. 2. Solve optimization problem |
If it's tractable |
RL, Control Theory, Game Theory, Decision Theory |
Machine Learning, Electrical engineering, Operations Research, Economics (normative), Finance |

| Model human behavior to learn preferences and beliefs |
1. Collect data by observation or experiment. 2. Infer parameters and predict future behavior |
If it fits human data |
IRL, Econometrics (Structural Estimation) | Machine Learning, Economics (positive and behavioral), Political Science, Psychology/Neuroscience |
|=====

Don't want to over-emphasize this distinction. First, good models of human behavior may be approximations models that are tractable and so good for building approximate optimal decision makers. Second, with IRL there is an interesting interaction between the two problems. Because defining an appropriate utility function to optimize is hard, you infer a function from humans. Once you have it, you can use any technique you like to optimize it (and you could ultimately get performance on the task that's very different from human performance). 



Earlier chapters describe agent models for *optimal* decision making and planning in sequential decision problems (MDPs and POMDPs). These models can serve as a guide when we build an algorithm that makes good choices. If we can write down a utility function (which is not always easy -- see below) and define the relevant state and action spaces, then we can use the kind of agent models for MDPs and POMDPs described earlier (or approximations of those models). Creating algorithms that make good choices is a common application in AI and robotics, OR and engineering and in applied economics (e.g. algorithmic trading).

The previous chapter discussed the problem of learning the utility functions and beliefs of agents from their behavior. The MDP and POMDP agent models serve as *generative* models of behavior that can be inverted to learn about the agent. The models are a means to the end of modeling the agent's behavior. If the agent does not behave optimally, we'd be better off (all other things being equal) using a different model. 

Our main application for learning about agents is learning about human preferences. So for a given decision problem, we should ask whether human choices will be optimal (in the MDP or POMDP sense). If not, we might consider using an alternative generative model. One common alternative model has already been discussed: the agent with softmax noise. This model has been used in various IRL papers to deal with human deviations from optimality [cite: young dialogue turn-taking paper, zheng/zhang taxi paper]. The softmax model is analytically tractable and allows one to use existing MDP and POMDP solution approaches. Yet it has limited expressiveness. Softmax is a a model of *random* deviations from optimality. By averaging over multiple draws of the agent's behavior, the agent's optimal action can be directly inferred. (In other words: even without knowing $$\alpha$$ we still have an unbiased estimator of the agent's maximizing action. Likewise for other random noise models.) However humans may deviate from optimality *systematically*. That is, they may take a sub-optimal action (relative to their own preferences) with higher probability than the optimal action.[Examples of systematic errors: smoker smokes everyday while wanting to quit, a student making persistent errors on a series of exams that reflect some misunderstanding. Optical illusions.] In that case, we will need to consider generative models other than the softmax model.


### Human deviations from optimal action - Cognitive Bounds
Humans deviate from optimal behavior in sequential problems in myriad ways. Our focus is not on all such deviations. We focus on cases where deviations in MDP/POMDP problems are both substantial (such that they significantly alter preferences) and systematic (so cannot be captured by the softmax model or something simiar). For practical purposes, deviations will be less important if the relevant preferences and beliefs can be learned via some other method. 

A first class of deviation is cognitive or computational *bounds* (in the sense of *bounded rationality* or *bounded optimality*). Humans (and AI agents) will perform suboptimally on some MDPs and POMDPs due to basic computational constraints. For example, consider a POMDP where the agent learns about the locations of restaurants on a grid representing a large city and then navigates between them. The ideal POMDP agent will never forget a location once it has been observed. Yet a human would not recall the exact location of hundreds of restaurants from a single walk through a city. Moreover this forgetting will produce behavior that differs systematically from the optimal agent. 

Even simple POMDPs are intractable as they scale up in size [cite]. So for any finite agent there will be POMDPs that they cannot solve exactly. It's plausible that humans often encounter POMDPs of this kind. For example, in lab experiments humans make systematic errors in computationally trivial bandit problems [yu, michael lee papers --- see also the paper by finale and zoubin on humans in tiger and other basic POMDPs]. Thus it's likely that humans will make errors on more complex POMDPs. (Consider the example of designing scientific experiments. This problem has a long time horizon (e.g. a lab could perform hundreds of experiment in a year). The inference from a single experiment is challenging and combining experimental results is hard.) In a chapter in this section [link], we implement the kind of computational bound (or bias) that has been used to model actual human performance in bandit problems. 


### Human deviations from optimal action - Cognitive Biases
Cognitive bounds will apply to any finite agent. Humans also have cognitive *biases* that may be more idiosyncratic (and less likely to apply to artifical agents). There is a large literature on cognitive biases in psychology and behavioral economics [cite reviews]. One particularly relevant example are the biases summarized by "Prospect Theory" (including Loss Aversion and Framing Effects). These are biases in decisions between simple one-shot lotteries over money or other prizes. We'd expect them to lead to systematics errors in real-life scenarios modeled as (PO)MDPs (assuming people don't use computers for all their expected value calculations). Another important bias is *time inconsistency*. This bias has been used to explain addiction, procrastination, impulsive behavior and the use of pre-commitment. We describe the bias and implement time-inconsistent planning in the next chapter. 


### Learning preferences from bounded and biased agents
We've argued that humans have biases and bounds that are systematic (and so can't be captured by the softmax noise model) and relevant to their actions in (PO)MDPs. In order to infer human beliefs and preferences when these biases are present, we can extend our (PO)MDP agent generative in order to capture these biases. The next two chapters describe and implement models for time-inconsistent and myopic (near-sighted) planning. The final chapter in this section implements inference for agents with these biases. Here we show that these biases make a significant difference in terms of inferring preferences correctly. We also show that we can do inference over a large space of agents, by combining unbiased agents with various flavors of biased agent.


