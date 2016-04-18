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

Table 1 provides more detail about these two uses[^table]. The first chapters of the book focused on (1). We presented agent models for solving MDPs and POMDPs optimally. We demonstrated these models on toy problems (Gridworld and simple variants of Bandits) that are closely related to practical real-world problems. The previous [chapter](/chapters/4-reasoning-about-agents) (Chapter 4) by contrast, focused on (2). Chapter 4 used the MDP and POMDP agent models not to solve problems but as *generative models* of human behavior. This chapter discusses the limitations of using optimal (PO)MDP agent modes as generative models for (2). We argue that developing models of biased or bounded decision making (which might not help with (1)) will be valuable for (2).

[^table]: While (1) and (2) are distinct goals, there is significant overlap in the methods of achieving them. First, a good computational model of human behavior might be helpful in constructing an optimal decision maker. Second, the applied literature on IRL exhibits an interesting interaction between (1) and (2). A problem with (1) is that it's often hard to write down an appropriate utility function to optimize. The ideal utility function is one that reflects actual human preferences or at least "human preferences with respect to the particular practical problem". So by solving (2) we can solve one of the "key tasks" in (1). This is exactly the approach taken in various applications of IRL. <!-- Maybe say something about Safe AI and the task of altruistic government -->


TODO_daniel: Use markdown, raw html or an image to format this table. The column headers are "goal", "key tasks" and so on. There are two rows corresponding to (1) and (2) above. The cells in each row are separated by `<br>` tag. (The column headers are separated by vertical lines.) The rows contain lots of text and so they should cover multiple lines and have fairly small text. (While HTML is ideal, copying from Word/Pages or whatever would be fine). 

<hr><hr>
| Goal | Key tasks | Find optimal solution? | Subject Areas | Fields |
<hr><hr>
| Build optimal decision making systems | <br>
1. Define appropriate utility function and decision problem. 2. Solve optimization problem | <br>
If it's tractable | <br>
RL, Control Theory, Game Theory, Decision Theory | <br>
Machine Learning, Electrical engineering, Operations Research, Economics (normative), Finance | <br>

<hr>

| Model human behavior to learn preferences and beliefs | <br>
1. Collect data by observation or experiment. 2. Infer parameters and predict future behavior | <br>
If it fits human data | <br>
IRL, Econometrics (Structural Estimation) | <br>
Machine Learning, Economics (positive and behavioral), Political Science, Psychology/Neuroscience |
<hr><hr>



### Random vs. Systematic Errors
The agent models we've considered are models of *optimal* performance on (PO)MDPs. So if humans deviate from optimality on some (PO)MDP then these models won't predict human behavior well. It's important to recognize the flexibility of the optimal models. The agent can have any utility function and any initial belief distribution. We saw in the previous chapters that apparently irrational behavior can sometimes be explained in terms of inaccurate prior beliefs.

Yet certain kinds of human behavior resist explanation in terms of false beliefs or unusual preferences. Consider a cigarette smoker who smokes every day while wanting to quit. Such people have often quit previously and re-started multiple times. They can be well informed both about the health effects of smoking and the cravings they'll get from abstaining. It's hard to explain their persistent smoking in terms of inaccurate beliefs[^beliefs].

[^beliefs]: One could argue that the smoker has a temporary belief that smoking is high utility which causes them to smoke. This belief subsides after smoking a cigarette and is replaced with regret. To explain this in terms of a POMDP agent, there has to be an observation that triggers this change of belief via Bayesian belief-updating. But what is this observation? The smoker may have *cravings*. Yet cravings alter the smoker's desires or wants, rather than being observational evidence about the empirical world. 

One standard response is to model deviations from optimal behavior using softmax noise. As we've seen above, it's easy from a computational standpoint to move between perfect maximizing behavior and soft-max behavior. Various papers doing Inverse Reinforcement Learning (IRL) on human data take this approach refp:kim2014inverse and refp:zheng2014robust. Yet the softmax model also has limited expressiveness. It's a model of *random* deviations from optimal behavior. Bigger deviations from optimality are explained by more randomness overall. Models of random error might be a good fit for certain motor or perceptual tasks (e.g. throwing a ball or locating the source of a distant sound). But the smoking example suggests that humans deviate from optimality *systematically*. That is, when not behaving optimally, humans actions remain predictable and bigger deviations from optimality do not imply more randomness.

Here are some examples of systematic deviations from optimal action:
<br>

>**Systematic deviations from optimal action**

- Smoking every week (i.e. systematically) while simultaneously trying to quit (e.g. using patches, throwing out cigarette packets).

- Always completing assignments just before the deadline, while always planning to complete the assignment as early as possible. 

- Forgetting random strings (passwords, ID numbers, words in a foreign language) within a few hours -- assuming they weren't explicitly memorized[^strings].

- Failing frequently on the same kind of math problem (e.g. long integer division).

[^strings]: With effort people can memorize these strings and store them in mind for longer periods. The claim is that if people do not make an attempt to memorize a random string, they will systematically forget the string within a short duration. This can't be easily explained on a POMDP model, where the agent has perfect memory.

These examples suggest that human behavior in everyday decision problems will not be easily captured by assuming softmax optimality. In the next sections, we divide these systematics deviations from optimality into *cognitive biases* and *cognitive bounds*. After explaining each category, we discuss their relevance to learning the preferences of agents. 


### Human deviations from optimal action: Cognitive Bounds

Humans perform sub-optimally on some MDPs and POMDPs due to basic computational constraints. Such constraints have been investigated in work on *bounded rationality* and *bounded optimality*, among other places. A simple example was mentioned above: people cannot quickly memorize random strings (even if the stakes are high). Similarly, consider the real-life version of our Restaurant Choice example. If you walk around a big city for the first time, you will forget the location of most of the restaurants you see on the way. If you try a few days later to find a restaurant, you are likely to take an inefficient route. This contrasts with the optimal POMDP-solving agent: he never forgets anything.

Limitations in memory are hardly unique to humans. For any AI system we might build today, there is some number of random bits that it cannot quickly place in permanent storage. So it makes sense to think about rational or optimal behave given various kinds of bounds on memory.  

Just as humans and machines have limited memory, they also have limited time. The simplest POMDPs, such as Bandit problems, are <a href="/chapters/3c-pomdp.html#complexity">intractable</a> -- the time needed to solve them will grow exponentially (or worse) in the problem size. The basic difficulty here is that optimal planning requires taking into account all possible sequences of actions and states. These explode in number as the number of states, actions, and possible sequences of observations grows[^grows].

[^grows]: Dynamic programming helps but does not tame the beast. There are many POMDPs that are small enough to be easily described (i.e. they don't have a very long problem description) but which we can't solve optimally -- even with the best computers. 

So for any agent with limited time there will be POMDPs that they cannot solve exactly. It's plausible that humans often encounter POMDPs of this kind. For example, in lab experiments humans make systematic errors in small POMDPs that are easy to solve with computers refp:zhang2013forgetful and refp:TODO:cite zoubin and finale. For much more complex real-world problems, humans use approximate heuristics and short-cuts to get around the time-complexity of the optimal solution.




### Human deviations from optimal action: Cognitive Biases
Cognitive bounds will apply to any finite agent. Humans also have cognitive *biases* that may be more idiosyncratic (and less likely to apply to artifical agents). There is a large literature on cognitive biases in psychology and behavioral economics [cite reviews]. One particularly relevant example are the biases summarized by "Prospect Theory" (including Loss Aversion and Framing Effects). These are biases in decisions between simple one-shot lotteries over money or other prizes. We'd expect them to lead to systematics errors in real-life scenarios modeled as (PO)MDPs (assuming people don't use computers for all their expected value calculations). Another important bias is *time inconsistency*. This bias has been used to explain addiction, procrastination, impulsive behavior and the use of pre-commitment. We describe the bias and implement time-inconsistent planning in the next chapter. 


### Learning preferences from bounded and biased agents
We've argued that humans have biases and bounds that are systematic (and so can't be captured by the softmax noise model) and relevant to their actions in (PO)MDPs. In order to infer human beliefs and preferences when these biases are present, we can extend our (PO)MDP agent generative in order to capture these biases. The next two chapters describe and implement models for time-inconsistent and myopic (near-sighted) planning. The final chapter in this section implements inference for agents with these biases. Here we show that these biases make a significant difference in terms of inferring preferences correctly. We also show that we can do inference over a large space of agents, by combining unbiased agents with various flavors of biased agent.


