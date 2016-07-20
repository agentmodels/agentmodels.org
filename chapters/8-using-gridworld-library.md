---
layout: chapter
title: How to use the WebPPL Agent Models library
description: Constructing various kinds of agents, how to write an MDP or POMDP, how to make your own Gridworld or k-arm bandit problem, how to call generic inference functions. Has codebox examples and links to the library's formal documentation. 
is_section: true
---


## Plan for guide

Goal of the guide is to make it easy for people to use the webppl-gridworld library. It should be self-contained, so that people don't need to go through all of agentmodels.org in order to find the guide useful. 

Contents:

1. Write an MDP (use line example from Section 3.1) and run MDP and hyperbolic agents. MDP has `transition` and `stateToAction`. For these agents, we could have a generic *simulateMDP* function, which takes an MDP (with `transition`), a startState, and an agent and computes the trajectory.

2. Write a POMDP. Could be line-world also: if state 1 says so, you go right, otherwise you go left. POMDP has `transition`, `beliefToAction`, `observation` functions. The startState will contain the latentState that agent is uncertain about. Work with `beliefDelay` agent to show comparison between optimal and boundVOI. Maybe discuss beliefAgent in footnotes. [Simulate should be flexible enough to implement some other kinds of agent. What about RL agents who don't know the transition or reward function?]

3. Gridworld MDP version. Show hiking example. Show how to vary the utilities. Run different agents on it. Show how to create variant gridworlds (need nicer interface for "feature").

4. Show how to create your own agent and run it on gridworld. Random agent. Epsilon-greedy agent instead of softmax. 

5. Bandits. Show how to create bandit problems. Run POMDP agents. Create your own POMDP agent.

-----------


### Introduction
This is a quick-start guide to using the webppl-agents library. For a more detailed, textbook-style explanation of the library, try [agentmodels.org](http://agentmodels.org). [Maybe provide a bit more info about what's included in agentmodels. e.g. mathematical background to the agents and basics of inference in webppl.]

The library is built around two basic entities: *agents* and *environments*. A *simulation* involves an agent interacting with a specific environment. We provide two standard RL environments as examples (Gridworld and Multi-armed Bandits). We provide four kinds of agent as examples. Many combinations of environment and agent are possible. In addition, it's easy to add your own environments and agents -- as we illustrate below. 

Environments and agents can't be freely combined. Among environments, we distinguish MDPs and POMDPs. For a POMDP environment, the agent must be a "POMDP agent", which means they maintain a belief distribution on the state. This separation of POMDPs and MDPs is not necessary, since POMDPs generalize MDPs. However, the separation makes the MDP code very short and perspicuous and also provides performance advantages. 

### Creating an MDP environment
We begin by creating a very simple MDP environment and running two agents from the library on that environment. 

MDPs are defined [here](http://agentmodels.org/chapters/3a-mdp.html). For use in the library, MDP environments are Javascript objects with `transition` and `stateToAction` methods. The `transition` method is a function from state-action pairs to states (as in the function $$T$$ in the MDP definition). The `stateToAction` method is a mapping from states to the actions that are allowed in that state.

MDPs are usually defined to include a *reward* or *utility* function on states or state-action pairs. We don't require that MDP objects include a utility function. But if we run an *agent* on the MDP, the agent must have a utility function defined on the states of the MDP. Since the same state space is used in both the MDP itself and the utility function, we often create the utility function when creating the MDP. 

Our first MDP environment is a simply a line where the agent can move left or right (starting from the origin). More precisely:

- *States:* Points on the integer line (e.g -1, 0, 1, 2).

- *Actions/transitions:* Actions “left”, “right” and “stay” move the agent deterministically along the line in either direction. We represent the actions as $$[-1,0,1]$$ in the code below. 

In our examples, the agent will start at the origin. The utility/reward function is as follows:

> `U(state)` is `1` when `state==0` (i.e. at the origin), `3` at `state==3` and `0` at all other states. 

The code for this MDP is straightforward. One additional requirement for an MDP object is that the transition function decrements the time and sets the `terminateAfterAction` attribute to `true` when the time has elapsed.

TODO: could use *environment* instead of *world*. 

~~~~
var advanceStateTime = function(state){
  var newTimeLeft = state.timeLeft - 1;
  return update(state, { 
    timeLeft: newTimeLeft,
    terminateAfterAction: newTimeLeft > 1 ? state.terminateAfterAction : true
  });
};


var makeLineMDP = function(totalTime){

    var stateToActions = function(state){return [-1, 0, 1];};

    var transition = function(state,action){
      var newLoc = state.loc + action;
      var stateNewLoc = update(state,{loc: newLoc});
      return advanceStateTime(stateNewLoc);
    };

    var world = {stateToActions:stateToActions, transition:transition};
    
    var startState = {timeLeft: totalTime, 
                      terminateAfterAction: false, 
                      loc: 0};

    var utility = function(state, action){    
      var table = {0:1, 3:3};
      return table[state.loc] ? table[state.loc] : 0;
    };

    return {world:world, startState:startState, utility:utility};
  };

wpEditor.put('makeLineMDP', makeLineMDP);
~~~~


We now create and simulate some agents from the library. [TODO: Explain how to construct agents, how to customize utility function]

~~~~
var makeLineMDP = wpEditor.get('makeLineMDP');

// Make world
var line = makeLineMDP(6);
var world = line.world;
var utility = line.utility;


// MDPAgent
var params = {alpha:1000, utility:utility};
var agent = makeMDPAgent(params, world);
var trajectory = simulate(line.startState, world, agent, 'states');
var locations = getLocations(trajectory);
assert.ok( last(locations)==3, 'MDPAgent test');

// random MDPAgent
var makeMDPAgentRandom = function(params, world){
  var stateToActions = world.stateToActions;
  var act = function(state){
    return Enumerate(function(){return uniformDraw(stateToActions(state));})
  };
  return {act:act, params:params};
  };
var agent = makeMDPAgentRandom(params, world);
var trajectory = simulate(line.startState, world, agent, 'states');
var locations = getLocations(trajectory);


// MDPAgentHyperbolic
var params = {alpha:1000,
              utility:utility,
              discount:2,
              sophisticatedOrNaive: 'naive'};
var agent = makeMDPAgent(params, world);
var trajectory = simulate(line.startState, world, agent, 'states');
var locations = getLocations(trajectory);
null
~~~~









