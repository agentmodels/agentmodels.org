
# Notes early chapters

# Chapter 1: Simulating agents and learning their preferences with probabilistic programs

## Motivating example
Animation of "donut temptation" case, maybe showing multiple agents (not just Bob).

Text: You have a dataset showing people's movements through a city. A single trajectory from this set is shown on a simplified map. Call the agent Bob. Given this trajectory, what can be infer about Bob's preferences and beliefs? First, we know Bob spent a long time at the Donut Store. So we infer he bought some food or drink there. But why did Bob not stop at the closer store to his starting point? He might just not know about Store 1. Maybe it just opened. Or he might know about it but prefer Store 2. Another possibility is that he intended to go the healthier "Salad box" above but ended up being tempted by the smell of Donuts.

In this tutorial, we'll build models of inference about the behavior of agents that can posit all these different explanations. These models can also simulate the behavior of different agents, allowing us to make predictions about what Bob would do if he was in a different part of town.

Suppose after seeing Bob, we saw a few more distinct individuals take exactly the same trajectory. How would this change our conclusions about Bob? Well, either *everyone* prefers Store 2 (because it's much better) or Store 1 just opened and so few people know about it. So it's more likely Bob doesn't know about D1 and less likely that he was tempted by the smell of Donuts. This kind of reasoning, where we assume different individuals have similar preferences or beliefs, will be formalized and simulated in later chapters.

## Agents as programs: making rational plans
Models of rational agents based on expected utility theory and Bayesian inference play an important role in the social and pschological sciences (as a model of human behavior) and in operations research, computer science and artificial intelligence (where they are used to create programs that are able to learn and act in an effective way). (Could add some citations). 

This tutorial implements utility-based (decision-theoretic) agents as [functional programs](wiki_FP_link). These programs provide a concise, intuitive translation of the mathematical specification of these agents into code. These agents simulate their own future plans via recursive calls. They update their beliefs by explicit Bayesian inference on their observations. 

Early chapters introduce agent models for classic sequential planning problems: MDPS and POMDPs. Example environments include discrete graphs, Gridworld, and multi-arm bandit problems. (Maybe exhibit some classic RL problem also?). The agents introduced here behave optimally (having full knowledge or perfect capacity to make optimal plans given partial knowledge). This assumption of optimal behavior won't always apply to humans. Chapter X introduces biased or bounded agents. The programs introduced for optimal agents can be slightly tweaked to model biased agents, illustrating the flexibility/extendability of our basic approach.

### Agents as programs: learning preferences from choices
Example 1 (above) illustrated a problem of learning or *inferring* an agent's preferences and beliefs from their behavior. The problem of learning people's preferences from observing their choices is important in economics ("revealed preference"), psychology and increasingly in machine learning and AI (recommender systems).

One approach to this problem is to model the agent using the utility-based models above, to model the environment as an MDP or POMDP, and then to infer the parameters that predict the observed behavior. This approach is called "structured estimation" in economics (cite something classic and something new), "inverse planning" in cognitive science, and "inverse reinforcement learning" (IRL) in machine learning / AI. It's been applied to infer the preferences of groups about health and education, to infer the preferences of drivers about how exactly to park a car, and to learn a nomadic groups preferences over areas of land.

Chapter X shows how to infer the preferences and beliefs of the agents we modeled in previous chapters. A great virtue of formulating agents as programs is that we can apply probabilistic programming techniques to carry out this inference automatically (without writing custom inference code). We illustrate full Bayesian inference and sampling-based approximations (rejection sampling, MCMC and particle filters). We also illustrate a standard non-Bayesian approach to IRL based on optimization (Abbeel and Ng). 


## Taster
This tutorial is about turning mathematical models of rational agents and turning them to programs for simulating plans and for learning preferences from observation. All the programs described run in your browser, along with rich visualizations of agent's taking actions and with graphs for displaying quantative features of our models and inferences. The language we used is Webppl, a probabilistic programming language embedded in Javascript. The next chapter will provide a quick introduction to Webppl before diving in to modeling agents. 

The box below is a taster of what you'll learn on the tutorial. This is an agent that solves the sequential planning problem in Example 1, preferring Store 2 to Store 1. The code runs live in the browser (with the encoding of the environment ommitted here -- we show later how to encode the environment). The math that the program implements is shown above.


# Chapter 2: Probabilistic programming in Webppl

## Introduction
This chapter introduces the probabilistic programming language Webppl (pronounced "web people") that we use to model agents throughout this tutorial. We give a brief overview of the features that are essential to this tutorial. If you have never encountered probabilistic programming before, you might benefit from reading some introductory material. There short articles [here](plenthusiast) and [here](mohammed) that provide a general overview. There is an interactive tutorial covering probabilistic programming and Bayesian inference at [probmods](https://probmods.org), which uses a language very similar to Webppl. If you have some background in programming languages, there is a [tutorial](https://dippl.org) on how to implement Webppl (which will also give a sense of how the language works).

Most of the code examples we provide will run in your browser (ideally Chrome/Safari). Webppl can also be installed locally and run from the command line --- see [here](https://webppl.org).


## Webppl: a functionally pure subset of Javascript
Webppl includes a subset of Javascript, and follows the syntax of Javascript for this subset. (Since we only use a limited subset of Javascript, you will only need a basic knowledge of Javascript to use Webppl). 

This program uses most of the available JS syntax:

~~~~
// Warn user when input to natural log function not positive
var verboseLog = function(x) {
    var warning = "Warning: input to log was not a positive number"
    var testCondition = (typeof(x) != 'number') || x <= 0
    if (testCondition){
        console.log(warning)}
    return Math.log(x)
}

[verboseLog(1), verboseLog(-1), verboseLog({}), verboseLog(true)] 

~~~~

Essential features:
Language is subset of JS. (Since we only use small subset, you'll be fine if you only know very basics of JS). 

Only functional things, or library calls to non-functional (provided they don't take webppl functions or violate referential opacity) JS.

examples:
can't use assignment. can't use for loop. use copying and map.

libraries: we will use math library and webppl builtins. 



webppl primitives
erps. bernoulliERP (flip). uniformDraw. normal.

attributes of erps.

definition of geometric, use recurison. then show properties of ERP.

inference:
enumerate, condition, factor. rejection / particle filter. 


