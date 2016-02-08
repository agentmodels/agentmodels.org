---
layout: chapter
title: Introduction
description: "Practical example of inferring human preferences from observed choices. Implementing agent model from decision theory as functional programs. Inferring preferences (IRL) by inverting agent models."
is_section: true
---

## Example: Learning preferences and beliefs from behavior

![Donut temptation gridworld](/assets/img/ch1_donut_new.png)


Suppose we have a dataset which records how individuals move through a city. The figure above shows what a datapoint from this set might look like. It depicts an individual, who we'll call Bob, moving along a street and then dwelling in the location of a restaurant. This restaurant is one of two nearby branches of a chain of Donut Stores. Two other nearby restaurants are also shown on the map.

From the evidence of Bob's movements alone, what can we infer about Bob's preferences and beliefs? Since Bob spent a long time at the Donut Store, we infer he bought some food or drink there. Since Bob could easily have walked to the other nearby eateries, we infer that Bob has a preference for donuts over noodles or salad.

Assuming Bob does like donuts, why did he not choose the store closer to his starting point ("Donut South")? The cause might be Bob's *beliefs* rather than his *preferences*. Maybe he does not know about "Donut South" because it just opened. Maybe Donut South has different hours than Donut North and Bob knows about this.

A different explanation is that Bob *intended* to go the healthier "Vegetarian Salad Bar". However, the most efficient route to the Salad Bar took him right to Donut North and once standing right outside he suddenly found the donuts more tempting than salad.

We've described a variety of inferences about Bob which would explain his behavior. This tutorial develops models for inference that can consider all of these different explanations and quantitatively compare their plausibility. These models can also simulate an agent's behavior in novel scenarios: we could predict Bob's behavior if he had started looking for food in a different part of the city. 

After consider the data about Bob, suppose the dataset showed that a a significant number of different individuals took exactly the same path as he did. How would this change our conclusions about Bob? It could be that everyone is tempted away from healthy food in the way Bob was. But this seems unlikely. Instead, it's now more plausible that Donut South is closed or that it's a brand new branch that few people know about. 

This kind of reasoning, where we make assumptions of the distributions of beliefs within populations, will be formalized and simulated in later chapters. We'll also consider multi-agent behavior where coordination or competition are important. 


## Agents as programs

### Making rational plans



Models of rational agents based on expected utility theory and Bayesian inference play an important role in the social and pschological sciences (as a model of human behavior) and in operations research, computer science and artificial intelligence (where they are used to create programs that are able to learn and act in an effective way). (Could add some citations). 

This tutorial implements utility-based (decision-theoretic) agents as [functional programs](wiki_FP_link). These programs provide a concise, intuitive translation of the mathematical specification of these agents into code. These agents simulate their own future plans via recursive calls. They update their beliefs by explicit Bayesian inference on their observations. 

Early chapters introduce agent models for classic sequential planning problems: MDPS and POMDPs. Example environments include discrete graphs, Gridworld, and multi-arm bandit problems. (Maybe exhibit some classic RL problem also?). The agents introduced here behave optimally (having full knowledge or perfect capacity to make optimal plans given partial knowledge). This assumption of optimal behavior won't always apply to humans. Chapter X introduces biased or bounded agents. The programs introduced for optimal agents can be slightly tweaked to model biased agents, illustrating the flexibility/extendability of our basic approach.

### Learning preferences from choices

Example 1 (above) illustrated a problem of learning or *inferring* an agent's preferences and beliefs from their behavior. The problem of learning people's preferences from observing their choices is important in economics ("revealed preference"), psychology and increasingly in machine learning and AI (recommender systems).

One approach to this problem is to model the agent using the utility-based models above, to model the environment as an MDP or POMDP, and then to infer the parameters that predict the observed behavior. This approach is called "structured estimation" in economics (cite something classic and something new), "inverse planning" in cognitive science, and "inverse reinforcement learning" (IRL) in machine learning / AI. It's been applied to infer the preferences of groups about health and education, to infer the preferences of drivers about how exactly to park a car, and to learn a nomadic groups preferences over areas of land.

Chapter X shows how to infer the preferences and beliefs of the agents we modeled in previous chapters. A great virtue of formulating agents as programs is that we can apply probabilistic programming techniques to carry out this inference automatically (without writing custom inference code). We illustrate full Bayesian inference and sampling-based approximations (rejection sampling, MCMC and particle filters). We also illustrate a standard non-Bayesian approach to IRL based on optimization (Abbeel and Ng). 

## Taster

This tutorial is about turning mathematical models of rational agents into programs for simulating plans and for learning preferences from observation. The programs all run in the browser, accompanied by visuals showing the agent's actions. The language of the tutorial is WebPPL, a probabilistic programming language based on Javascript refp:dippl. As a taster, here is a simple code snippet in WebPPL, using the interactive code boxes that we'll use throughtout. 

~~~~
var coinFlip = function(){return flip() ? 'H' : 'T';};
print("Some coin flips:")
print(repeat(5, coinFlip))

var geometric = function(p) {
  return flip(p) ? 1 + geometric(p) : 1
};
geometric(0.5);
~~~~

The [next chapter](/chapters/02-agents-as-models.html) provides an introduction to WebPPL. 


## References

- cite:dippl
