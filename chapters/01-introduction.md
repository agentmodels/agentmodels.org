---
layout: chapter
title: Introduction
description: "Motivating example of inferring preferences from observed choices. Introduce utility-based, decision-theoretic agents implementing as programs. Introduce inference of preferences by inverting the agent models. Taster of a Webppl agent model"
is_section: true
---

## Motivating example
![picture](https://github.com/agentmodels/agentmodels.org/blob/gh-pages/assets/img/ch1_donut_tempt_small.png)


[Eventually: Animation of "donut temptation" case, maybe showing multiple agents (not just Bob).]

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

<!-- simple example of code box that's easy to read for everyone -->
<!-- WebPPL, ADD PICTURE OF BOB'S PATH -->

The box below is a taster of what you'll learn on the tutorial. This is an agent that solves the sequential planning problem in Example 1, preferring Store 2 to Store 1. The code runs live in the browser (with the encoding of the environment ommitted here -- we show later how to encode the environment). The math that the program implements is shown above.

[add math]


~~~~
var agent = cache(
  function(_agent, _expUtility, state, timeLeft, params){
    return Enumerate(function(){
      var action = uniformDraw(params.actions);
      var eu = _expUtility(_agent, _expUtility, state, action, timeLeft, params);    
        factor(params.alpha * eu);
        return action;
      });
    }
  );
  
  var expUtility = cache(
    function(_agent, _expUtility, state, action, timeLeft, params){
      var utility = params.utility;
      var u = utility(state,action);
      
      if (timeLeft - 1 == 0){
        return u;
      } else {                     
        return u + expectation( Enumerate(function(){
          var transition = params.transition;
          var nextState = transition(state, action); 
          var nextAction = sample(_agent(_agent, _expUtility, nextState, timeLeft-1, params));
          return _expUtility(_agent, _expUtility, nextState, nextAction, timeLeft-1, params);  
        }));
      }                      
    });

var simulate = function(startState, actualTotalTime, perceivedTotalTime, params, output){

    var sampleSequence = function(state, actualTimeLeft, perceivedTimeLeft){
      if (actualTimeLeft==0){
        return [];
      } else {
        var action = sample(agent(agent, expUtility, state, perceivedTimeLeft, params));
        var transition = params.transition;
        var nextState = transition(state,action);
        var outputItem = output == 'actions' ? action : state;
        return [ outputItem ].concat( sampleSequence(nextState,actualTimeLeft-1, perceivedTimeLeft-1));
      }
    };
    // could sample sequence without enumerating by returning *sampleSequence(args)*
    return numberRejectionSamples==0 ? Enumerate(function(){
      return sampleSequence(startState, actualTotalTime, perceivedTotalTime); 
    }) : 
    Rejection(function(){
      return sampleSequence(startState, actualTotalTime, perceivedTotalTime); 
    }, numberRejectionSamples, undefined, true);
  };
  
~~~~
