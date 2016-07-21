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

MDPs are defined [here](http://agentmodels.org/chapters/3a-mdp.html). For use in the library, MDP environments are Javascript objects with the following methods:

>`var myMDPEnvironment = {transition: ...,  stateToActions: ...}`

The `transition` method is a function from state-action pairs to states (as in the function $$T$$ in the MDP definition). The `stateToAction` method is a mapping from states to the actions that are allowed in that state.

MDPs are usually defined to include a *reward* or *utility* function on states or state-action pairs. We don't require that MDP objects include a utility function. But if we run an *agent* on the MDP, the agent must have a utility function defined on the states of the MDP. Since the same state space is used in both the MDP itself and the utility function, we often create the utility function when creating the MDP. 

#### The Line MDP environment
Our first MDP environment is a simply a line where the agent can move left or right (starting from the origin). More precisely, the Line MDP is as follows:

- **States:** Points on the integer line (e.g ..., -1, 0, 1, 2, ...).

- **Actions/transitions:** Actions “left”, “right” and “stay” move the agent deterministically along the line in either direction. We represent the actions as $$[-1,0,1]$$ in the code below. 

In our examples, the agent's `startState` is the origin. The utility is 1 at the origin, 3 at the third state right of the origin ("state 3"), and 0 otherwise.

The transition function must also decrements the time. States are objects with a `terminateAfterAction` attribute, which terminates the MDP when true. In the example below, `terminateAfterAction` is set to `true` when the state's `timeLeft` attribute is set to 1. For the Line MDP, an example state (the `startState`) has form:

>`{terminateAfterAction: false, timeLeft:6, loc:0}`

TODO: could say "environment" instead of "world" in the code below. but we used used "world" almost everywhere in the library. 

~~~~
// helper function that decrements time and triggers termination when time elapsed
var advanceStateTime = function(state){
  var newTimeLeft = state.timeLeft - 1;
  return update(state, { 
    timeLeft: newTimeLeft,
    terminateAfterAction: newTimeLeft > 1 ? state.terminateAfterAction : true
  });
};

// constructuor for the "line" MDP environment:
// argument *totalTime* is the time horizon
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

To run an agent on this MDP, we use a `makeAgent` constructor and the library function `simulate`. The constructor for MDP agents is `makeMDPAgent`:

>`makeMDPAgent(params, world)`

For an optimal (non-discounting) agent, the parameters are:

>`{utility: <utility_function>,  alpha: <softmax_alpha>}`

Agent constructors always have these same two arguments. The `world` argument is required for the agent's internal simulations of possible transitions. The `params` argument specifies the agent's parameters and whether the agent is optimal or biased.

An environment (or "world") and agent are combined with the `simulate` function:

>`simulate(startState, world, agent, outputType)`

Given the utility function defined above, the highest utility state is at location 3 (three steps to the right from the origin). So a non-discounting agent will move and stay at this location. 

~~~~

///fold:
// helper function that decrements time and triggers termination when time elapsed
var advanceStateTime = function(state){
  var newTimeLeft = state.timeLeft - 1;
  return update(state, { 
    timeLeft: newTimeLeft,
    terminateAfterAction: newTimeLeft > 1 ? state.terminateAfterAction : true
  });
};

// constructuor for the "line" MDP environment:
// argument *totalTime* is the time horizon
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
///

// Construct line MDP environment
var totalTime = 6;
var lineMDP = makeLineMDP(totalTime);
var world = lineMDP.world;

// The lineMDP object also includes a utility function and startState
var utility = lineMDP.utility;
var startState = lineMDP.startState;


// Construct MDP agent
var params = {alpha:1000, utility:utility};
var agent = makeMDPAgent(params, world);

// Simulate the agent on the lineMDP with *outputType* set to *states*
var trajectory = simulate(startState, world, agent, 'states');

// Display start state 
print(trajectory)

~~~~



More agents:

~~~~

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









