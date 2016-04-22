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

Table 1 provides more detail about these two uses[^table]. The first chapters of the book focused on (1). We presented agent models for solving MDPs and POMDPs optimally. We demonstrated these models on toy problems (Gridworld and simple variants of Bandits) that are closely related to practical real-world problems. Chapter 4, by contrast, focused on (2). (PO)MDP agent models were used not to solve practical problems but as *generative models* of human behavior. The present chapter discusses the limitations of using optimal (PO)MDP agent modes as generative models for (2). We argue that developing models of *biased* or *bounded* decision making will be valuable for (2).

<img src="/assets/img/table_chapter5_intro.png" alt="table" style="width: 650px;"/>

>**Table 1:** Two uses for formal models of sequential decision making. The heading "Optimality" means "Are optimal models of decision making used?".
<br>

[^table]: While (1) and (2) are distinct goals, there is significant overlap in the methods of achieving them. First, a good computational model of human behavior might be helpful in constructing an optimal decision maker. Second, the applied literature on IRL exhibits an interesting interaction between (1) and (2). A challenge with (1) is that it's often hard to write down an appropriate utility function to optimize. The ideal utility function is one that reflects actual human preferences (at least as they pertain to the particular problem). So by solving (2) we can solve one of the "key tasks" in (1). This is exactly the approach taken in various applications of IRL. <!-- Maybe say something about Safe AI and the task of altruistic government -->

<!-- TABLE. TODO: find nice html/markdown rendering:

Goal|Key tasks|Optimality?|Sub-fields|Fields
Solve practical decision problems|1. Define appropriate utility function and decision problem.
 
2. Solve optimization problem|If it’s tractable|RL, Game and Decision Theory, Experimental Design|ML/Statistics, Operations Research, Economics (normative)
Learn the preferences and beliefs of humans|1. Collect data by observation/experiment.

2. Infer parameters and predict future behavior|If it fits human data|IRL, Econometrics (Structural Estimation), Inverse Planning|ML, Economics (positive),
Psychology, Neuroscience
-->



### Random vs. Systematic Errors
The agent models we've considered are models of *optimal* performance on (PO)MDPs. So if humans deviate from optimality on some (PO)MDP then these models won't predict human behavior well. It's important to recognize the flexibility of the optimal models. The agent can have any utility function and any initial belief distribution. We saw in the previous chapters that apparently irrational behavior can sometimes be explained in terms of inaccurate prior beliefs.

Yet certain kinds of human behavior resist explanation in terms of false beliefs or unusual preferences. Consider the following:

>**The Smoker** <br> Fred smokes cigarettes every day. He has tried to quit multiple times and still wants to quit. He is fully informed about the health effects of smoking and of the cravings that make quitting difficult. 

It's hard to explain this persistent smoking in terms of inaccurate beliefs[^beliefs]. Similar behavior is seen in problem gamblers and in other compulsions.

[^beliefs]: One could argue that Fred has a temporary belief that smoking is high utility which causes him to smoke. This belief subsides after smoking a cigarette and is replaced with regret. To explain this in terms of a POMDP agent, there has to be an *observation* that triggers the belief-change via Bayesian updating. But what is this observation? Fred has *cravings*, but these cravings alter Fred's desires or wants, rather than being observational evidence about the empirical world. 

One standard response is to model deviations from optimal behavior using softmax noise. Various papers doing Inverse Reinforcement Learning (IRL) on human data take this approach refp:kim2014inverse and refp:zheng2014robust. Yet the softmax model also has limited expressiveness. It's a model of *random* deviations from optimal behavior: larger deviations from optimality entail larger amounts of randomness.

Models of random error might be a good fit for certain motor or perceptual tasks (e.g. throwing a ball or locating the source of a distant sound). But the smoking example suggests that humans deviate from optimality *systematically*. That is, when not behaving optimally, humans actions remain *predictable* and larger deviations from optimality do not imply larger amounts of randomness.

Here are some examples of systematic deviations from optimal action:
<br>

>**Systematic deviations from optimal action**

- Smoking every week (i.e. systematically) while simultaneously trying to quit (e.g. using patches, throwing out cigarette packets).

- Always completing assignments just before the deadline, while always planning to complete the assignment as early as possible. 

- Forgetting random strings (passwords or ID numbers) within a few hours -- assuming they weren't explicitly memorized[^strings].

- Making mistakes on arithmetic problems[^math] (e.g. long division).

[^strings]: With effort people can memorize these strings and store them in mind for longer periods. The claim is that if people do not make an attempt to memorize a random string, they will systematically forget the string within a short duration. This can't be easily explained on a POMDP model, where the agent has perfect memory.

[^math]: People learn the algorithm for long division but still make mistakes -- even when stakes are relatively high (e.g. important school exams). While humans vary in their math skill, all humans have severe limitations (compared to computers) at doing arithmetic. See refp:dehaene2011number for various robust, systematic limitations in human numerical cognition. 

These examples suggest that human behavior in everyday decision problems will not be easily captured by assuming softmax optimality. In the next sections, we divide these systematics deviations from optimality into *cognitive biases* and *cognitive bounds*. After explaining each category, we discuss their relevance to learning the preferences of agents. 


### Human deviations from optimal action: Cognitive Bounds

Humans perform sub-optimally on some MDPs and POMDPs due to basic computational constraints. Such constraints have been investigated in work on *bounded rationality* and *bounded optimality* refp:gershman2015computational. A simple example was mentioned above: people cannot quickly memorize random strings (even if the stakes are high). Similarly, consider the real-life version of our Restaurant Choice example. If you walk around a big city for the first time, you will forget the location of most of the restaurants you see on the way. If you try a few days later to find a restaurant, you are likely to take an inefficient route. This contrasts with the optimal POMDP-solving agent: he never forgets anything.

Limitations in memory are hardly unique to humans. For any current autonomous robot, there is some number of random bits that it cannot quickly place in permanent storage. In addition to constraints on memory, humans and machines have constraints on time. The simplest POMDPs, such as Bandit problems, are <a href="/chapters/3c-pomdp.html#complexity">intractable</a>: the time needed to solve them will grow exponentially (or worse) in the problem size refp:cassandra1994acting,  refp:madani1999undecidability. The issue is that optimal planning requires taking into account all possible sequences of actions and states. These explode in number as the number of states, actions, and possible sequences of observations grows[^grows].

[^grows]: Dynamic programming helps but does not tame the beast. There are many POMDPs that are small enough to be easily described (i.e. they don't have a very long problem description) but which we can't solve optimally -- even with the best computers. 

So for any agent with limited time there will be POMDPs that they cannot solve exactly. It's plausible that humans often encounter POMDPs of this kind. For example, in lab experiments humans make systematic errors in small POMDPs that are easy to solve with computers refp:zhang2013forgetful and refp:doshi2011comparison. Real-world tasks with the structure of POMDPs, such as choosing how to invest resources or deciding on a sequence of scientific experiments, are much more complex and so presumably can't be solved by humans exactly.

### Human deviations from optimal action: Cognitive Biases

Cognitive bounds of time and space (for memory) mean that any realistic agent will perform sub-optimally on some problems. By contrast, the term "cognitive biases" is usually applied to errors that are idiosyncratic to humans and would not arise in AI systems[^biases]. There is a large literature on cognitive biases in psychology and behavioral economics refp:kahneman2011thinking, refp:kahneman1984choices. One relevant example is the cluster of biases summarized by *Prospect Theory* refp:kahneman1979prospect. In one-shot choices between "lotteries", people are subject to framing effects (e.g. Loss Aversion) and to erroneous computation of expected utility[^prospect]. Another important bias is *time inconsistency*. This bias has been used to explain addiction, procrastination, impulsive behavior and the use of pre-commitment devices. We describe and implement time-inconsistent agents in the next chapter. 

[^biases]: We do not presuppose a well substantiated scientific distinction between cognitive bounds and biases. Many have argued that biases result from heuristics and that the heuristics are a fine-tuned shortcut for dealing with cognitive bounds. For our purposes, the main distinction is between intractable decision problems (such that any agent will fail on large enough instances of the problem) and decision problems that appear trivial for simple computational systems but hard for some proportion of humans. For example, time-inconsistent behavior appears easy to avoid for computational systems but hard to avoid for humans. 

[^prospect]: The problems descriptions are extremely simple. So this doesn't look like an issue of bounds on time or memory forcing people to use a heuristic or approximate approach. 


### Learning preferences from bounded and biased agents
We've asserted that humans have cognitive biases and bounds. These lead to systemtic deviations from optimal performance on (PO)MDP decision problems. As a result, the softmax-optimal agent models from previous chapters will not always be good generative models for human behavior. To learn human beliefs and preferences when such deviations from optimality are present, we extend and elaborate our (PO)MDP agent models to capture these deviations. The next chapter implements time-inconsistent agents via hyperbolic discounting. The subsequent chapter implements "greedy" or "myopic" planning, which is a general strategy for reducing time- and space-complexity. In the final chapter of this section, we show (a) that assuming humans are optimal can lead to mistaken inferences in some decision problems, and (b) that our extended generative models can avoid these mistakes.

---------

### Footnotes
