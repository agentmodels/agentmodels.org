---
layout: chapter
title: "Partial observability"
description: Mathematical framework, implementation in WebPPL, Gridworld and restaurants example, bandit problems.
---

<!--
PLAN

1. Restaurant example. Human might not know which restaurants open or full. Might not know about certain restaurants. Might not know whether the restaurants have good food or have good food often. Intuition: move around world, get observatinos from world. Update beliefs based on observations, enabling better choices. Working backwards, when simulating sequences of actions, take into account possible observations and better decisions that result -- i.e. VOI of certain actions. So need to model one's future belief states. 

2. Formalizing this. Basic structure same as in MDP gridworld. Only change is an observation function that gives a dist on observations as a function of the state (i.e. agent's position). From high level, formalize as agent being uncertain about which state it is in. For instance, it might be uncertain about its location in the world and uncertain about whether a restaurant is current open or closed. Agent then receives obserations. E.g. it gets to observe its location (even if location depends on stochastic transitions). It observes whether a restaurant is open if it gets close enough. Formalization in AAAI paper. Update beliefs by simulating world given different starting states. (note. if the agent always gets to observe their state, then you are wasting time doing the belief update. you might as well just call them on the state. if there is uncertainty about location, which is realistic, then you get a slam style problem. you can easily get less and less confident about where you are as you take more actions. so then you can't condition on (manifestState,observation) pair because you don't know your manifestState. examlple: maybe different locations on grid have unknown reward distributions. you end up observing these rewards. but you don't know where you are on the grid. so if you then get shifted to somewhere else on the grid, you don't know where to go to get the reward you previously observed.



3. More specialized formulation. Assume that agent faces an MDP parameterized by some variables whose values are unknown. These are variables are fixed for any given decision problem. For example, whether a restaurant is present in a particular location is fixed. The mean quality of a restaurant is fixed, etc. We call the vector of these variables the latent state. Given the latent state, the MDP has its normal state, which in gridworld is the agent's location, and we have the current observation. The agent's current location is assumed to be known without any observations. We call this kind of state the manifest state. Agent program, agent is called on a manifest state, prior distribution and obseration. Computes exp U of actions by simulating future state sequences given different possible latent states, with dist on latent states the posterior given the current observation. To simulate the world containing this agent, the world takes a state ==df {manifestState: [0,1], latentState:{donutSouthOpen: false, noodleShopOpen: true}}. The transition function and utility function depend on both the manifest and latentState. For example, if a restaurant exists, it will yield some utility and be terminal. Observations also depend on both. -->
 
## Introduction: Agents with uncertainty and belief updating [WORK IN PROGRESS]

In the stochastic MDPs of previous chapters, the agent is uncertain about the result of taking an action. In the Gridworld Hiking example, Alice is uncertain whether she would fall down the hill if she takes the shortcut. In MDPs, this uncertainty cannot be reduced by observation: it's assumed to be intrinsic to the environment. In contrast, humans are often uncertain about features of the environment that are *observable*. Realistically, Alice would observe the weather conditions and update her beliefs dynamically about how risky the shortcut would be. In the example of Bob choosing between restaurants, Bob would not have complete knowledge of the restaurants in his neighborhood. He'd be uncertain about opening hours, chance of getting a table, restaurant quality, the exact distances between locations, and so on. Yet this uncertainty could be easily reduced by observation. 

## Extending our agent model for POMDPs
[WORK IN PROGRESS]
We now relax the assumption that the agent knows the true world state. Instead, we use a distribution $$p(s)$$ to represent the agent's belief about which state holds. Using a likelihood function $$p(o|s)$$, the agent can update this belief.

[ADD: Formal definition of POMDP agent].

We present a simple extension of our previous agent model for MDPs. The agent is uncertain about the world state. For instance, the agent may be uncertain about whether a restaurant is open or closed. For a bandit problem, the agent is uncertain about the expected value of a particular arm. As before, we assume the agent knows some of the state without observation. In Gridworld, the agent knows their location in the grid. In Bandits, the agent knows which arm they just pulled. We call the unknown state the `latentState` and the known state the `manifestState`. The `agent` function is called on a `manifestState`, updates beliefs about the `latentState`, and computes expectations for actions (integrating over posterior uncertainty in the `latentState`). The `simulate` function operates on `states`, where the state-space is simply the Cartesian product of the manifest and latent state spaces, e.g. in the Gridworld domain, an example state would have the form:

`state == {manifestState: [0,1], latentState: { donutRestaurantOpen: true } }`

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
