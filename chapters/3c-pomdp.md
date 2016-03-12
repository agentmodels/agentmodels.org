---
layout: chapter
title: "Partial observability"
description: Mathematical framework, implementation in WebPPL, Gridworld and restaurants example, bandit problems.
---

Use math formalism of the paper and from Kaelbling et al paper. Introduce simplified version of beliefAgent.wppl. Bring updateBelief into scope of agent. Simplify simulate. Might be worth discussing recursing on state vs. belief but not clear.

Should we introduce bandit example here? Yes, because we'll want to talk about it for myopic and boundVOI agents and it's good to have multiple examples. If so, showing stochastic bandits also would be ideal -- otherwise we have no stochasticity in the environment for the next few chapters. This also is a good way to introduce the intractability of POMDPs.

 
## Introduction: Agents with uncertainty and belief updating [WORK IN PROGRESS]

The previous chapters included MDPs where the transition function is *stochastic*. This means the agent is *uncertain* about the result of taking an action in a given state. For example in Gridworld Hiking, Alice is uncertain whether she would fall down the hill if she takes the shortcut. In an MDP the agent's uncertainty cannot be altered by observation. Transitions occur according to a particular probability distribution that is fixed (with no learnable parameters). An MDP is like a fair lottery: observing the winning ticket one week does not change the distribution on tickets the following week.  

In contrast, we often face problems where our uncertainty can be *reduced* by observation. In the example of Bob choosing between restaurants, Bob would not have complete knowledge of the restaurants in his neighborhood. He'd be uncertain about opening hours, chance of getting a table, restaurant quality, the exact distances between locations, and so on. This uncertainty can be reduced observation: Bob can walk to the restaurant and see whether or not it's open. In other examples, the environment is stochastic but the agent can gain knowledge of the *distribution* on outcomes. For example, in Multi-arm Bandit problems, the agent learns a distribution over rewards for some of the arms by observing their rewards. 


## Extending our agent model for POMDPs

- the environment now includes an observation function from states to observations.
- agent has prior uncertainty about some elements of the environment. these elements could influence observations, transitions or utilities. we focus on the case where they influence observations and transitions.
- in our examples, apart from the observation function, the environment has the same structure as before (including the markov assumption). previously an agent was given a state as input and had to take an action (which caused a transition). now the agent is uncertain about which state they are in. consider the example of Restaurant Choice where Bob doesn't know whether a restaurant is open or not. the property "restaurant is open" can be thought of as part of the state. Another aspect is Bob's location. If Bob knows his location but doesn't know if Donut South is open, then he has a distribution over the states [{myLocation:[2,1], donutSouth:'closed'}, {myLocation:[2,1], donutSouth:'open'}]. Bob might also be uncertain about his location (esp. if he's out at night in unfamiliar city) but we won't consider that example in the seqel. 


~~~~

// generalization of *mdpSimulate* from previous chapters

var pomdpSimulate = function(startState, actualTotalTime, 
                              perceivedTotalTime, params){

  // Key functions defining POMDP
  var utility = params.utility;
  var transition = params.transition;
  var observe = params.observe;
  var observationEquality = params.observationEquality;

  // Constructor for states
  var buildState = function(manifestState,latentState){
    return {manifestState:manifestState, latentState:latentState};
  };

  // Takes agent's belief ERP and updates it on a single observation.
  // Since *observe* takes a state (not a latentState) 
  // we need to build a state from the sampled latentState. 
  var updateBelief = dp.cache(
    function(currentBelief, manifestState, observation, params){

      return Enumerate( function(){
        var hypotheticalLatentState = sample(currentBelief);
        var hypotheticalState = buildState(manifestState, hypotheticalLatentState);
        var hypotheticalObservation = observe(hypotheticalState, params);
        condition( observationEquality(hypotheticalObservation, observation) );
        return hypotheticalLatentState;
      });
    });
  

  // Agent is called on *manifestState* not *state* 
  // since he doesn't know his *state*
  
  var agent = dp.cache( 
    function(manifestState, timeLeft, currentBelief, observation, params){
    
    return Enumerate( function(){
      var updatedBelief = updateBelief(currentBelief, manifestState,
                                       observation);
      var action = uniformDraw(params.actions);
      
      var expectedUtility = expectation(
        Enumerate(function(){
          var state = buildState(manifestState, sample(updatedBelief));
          return expUtility(state, action, timeLeft, updatedBelief, params);   
        }));
      
      factor(params.alpha * expectedUtility);
      return {action: action, belief: updatedBelief};
    });
  });
  
  
  var expUtility = dp.cache(
    function(state, action, timeLeft, currentBelief, params){ 
      var u = utility(state, action, params);
      
      if (timeLeft - 1 == 0){
        return u;
      } else {                     
        return u + expectation( Enumerate(function(){
          var nextState = transition(state, action, params);
          var nextManifestState = getManifestState(nextState,params);
          var observation = observe(nextState,params);
          var out = sample(agent(nextManifestState, timeLeft-1, 
                                 currentBelief, observation, params));
          var nextAction = out.action;
          var nextBelief = out.belief;
          return expUtility(nextState, nextAction, timeLeft-1, nextBelief, params);
        }));
      }                      
    });
  

  var simulate = function(startState, actualTotalTime, perceivedTotalTime, params){
    
    var sampleSequence = function(state, actualTimeLeft, perceivedTimeLeft, 
                                   history, currentBelief, observation){
      
      if (actualTimeLeft==0){
        return history.slice(0,history.length-1);
      } else {
        
        var out = sample(agent(getManifestState(state, params), perceivedTimeLeft,
                               currentBelief, observation, params));
        var action = out.action;
        var updatedBelief = out.belief;
        var nextState = transition(state, action, params);
        var nextObservation = observe(nextState, params);
        var nextHistory = push(history, nextState);
        
        return sampleSequence(nextState, actualTimeLeft-1, perceivedTimeLeft-1, 
                              nextHistory, updatedBelief, nextObservation);
      }
    };
    
    return Enumerate(function(){    
      var startHistory = [startState];
      var observation = observe(startState, params);
      var latentStatePrior = params.latentStatePrior;
      return sampleSequence(startState, actualTotalTime, perceivedTotalTime, 
                            startHistory, latentStatePrior, observation);
      
    });                 
  };
  
  return simulate(startState, actualTotalTime, perceivedTotalTime, params);
};
pomdpSimulate;
~~~~



--------------

[Table of Contents](/)
