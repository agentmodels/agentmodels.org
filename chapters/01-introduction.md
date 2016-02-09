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

After considering the data about Bob, suppose the dataset showed that a a significant number of different individuals took exactly the same path as he did. How would this change our conclusions about Bob? It could be that everyone is tempted away from healthy food in the way Bob was. But this seems unlikely. Instead, it's now more plausible that Donut South is closed or that it's a brand new branch that few people know about. 

This kind of reasoning, where we make assumptions of the distributions of beliefs within populations, will be formalized and simulated in later chapters. We'll also consider multi-agent behavior where coordination or competition are important. 


## Agents as programs

### Making rational plans

Formal models of rational agents play an important role in economics refp:rubinstein2012lecture and in the cognitive sciences refp:chater2003rational as models of human or animal behavior. Core components of such models are *expected-utility maximization*, *Bayesian inference*, and *game-theoretic equilibria*. These ideas are also applied in engineering and in artificial intelligence refp:russell1995modern in order to compute optimal solutions to problems or to construct artificial systems that learn and reason optimally. 

This tutorial implements utility-maximizing Bayesian agents as functional, probabilistic programs. These programs provide a concise, intuitive translation of the mathematical specification of rational agents into code. The implemented agents explicitly simulate their own future choices via recursion. They update beliefs by exact or approximate Bayesian inference. They reason about other agents by simulating them (which includes simulating the simulations of others). 

The first section of the tutorial implements agent models for sequential decision problems in stochastic environments. We introduce a program that solves finite-horizon and MDPs and show a simple extension to POMDPs. These agents behave *optimally*, making optimal plans given their knowledge of the world. Human behavior, by contrast, is often *sub-optimal*, whether due to irrational behavior or to constrained resources. The programs we use to implement optimal agents can, with slight modification, implement agents with biases (e.g. time inconsistency) and with resource bounds (e.g. bounded "look ahead" or Monte Carlo sampling).


### Learning preferences from behavior

The example of Bob (above) was not about simulating rational agents per se but about the problem *learning* or *inferring* an agent's preferences or beliefs from their choices. This problem is important to economics and psychology. Predicting preferences from past choices is also a major area of applied machine learning (e.g. for Netflix or for Facebook's newsfeed). 

One approach to this problem is to assume the agent is a rational utility-maximizer, to assume the environment is an MDP or POMDP, and to infer the utilities and beliefs and predict the observed behavior. This approach is called "structured estimation" in economics refp:aguirregabiria2010dynamic, "inverse planning" in cognitive science refp:ullman2009help, and "inverse reinforcement learning" (IRL) refp:ng2000algorithms in machine learning and AI. It has been applied to inferring the perceived rewards of education from observed work and education choices, preferences for health outcomes from smoking behavior, and the preferences of a nomadic group over areas of land (see cites in \refp:evans2015learning). 

[Section III](/chapters/07-reasoning-about-agents.md) shows how to infer the preferences and beliefs of the agents modeled in previous chapters. Since the agents were implemented as programs, we apply probabilistic programming techniques to perform inference with very little additional code. Inference techniques include exact Bayesian inference and sampling-based approximations (MCMC and particle filters).


## Taster

Our models of agents and of inference all run in "code boxes" in the browser, accompanied by animated visualizations agent behavior. The language of the tutorial is [WebPPL](https://webppl.org), an easy-to-learn probabilistic programming language based on Javascript refp:dippl. As a taster, here is a simple code snippet in WebPPL, using the interactive code boxes that we'll use throughtout. 

~~~~
// *flip* returns [true,false] with even probability
var coinFlip = function(){return flip() ? 'H ' : 'T ';};
print("Some coin flips:" + coinFlip() + coinFlip() + coinFlip());

// use *flip* to define a sampler for the geometric distribution
var geometric = function(p) {
  return flip(p) ? 1 + geometric(p) : 1
};

// produce a histogram from repeated sampling 
viz.print( repeat(100, function(){return geometric(0.8);}) );

~~~~

The [next chapter](/chapters/02-webppl.html) provides an introduction to WebPPL.

--------------

[Table of Contents](index.md)
