---
layout: chapter
title: Quick-start guide to the webppl-agents library
description: Create your own MDPs and POMDPs. Create gridworlds and k-armed bandits. Use agents from the library and create your own. 
is_section: true
---

<!--
## Plan for guide

Goal of the guide is to make it easy for people to use the `webppl-agents` library. It should be self-contained, so that people don't need to go through all of agentmodels.org in order to find the guide useful. 

Contents:

1. Write an MDP (use line example from Section 3.1) and run MDP and hyperbolic agents. MDP has `transition` and `stateToAction`.

3. Gridworld MDP version. Show hiking example. Show how to vary the utilities. Run different agents on it. Show how to create variant gridworlds (need nicer interface for "feature").

4. Show how to create your own agent and run it on gridworld. Random agent. Epsilon-greedy agent instead of softmax.

2. Write a POMDP. Could be line-world also: if state 1 says so, you go right, otherwise you go left. POMDP has `transition`, `beliefToAction`, `observation` functions. The startState will contain the latentState that agent is uncertain about. Work with `beliefDelay` agent to show comparison between optimal and boundVOI. Maybe discuss beliefAgent in footnotes.

5. Bandits. Show how to create bandit problems. Run POMDP agents. Create your own POMDP agent.

-->

### Contents

1. <a href="#intro">Introduction</a>

2. <a href="#createMDP">Creating MDPs</a>

3. <a href="#gridworld">Creating Gridworld MDPs</a>

4. <a href="#agents">Creating your own agents</a>

5. <a href="#createPOMDP">Creating POMDPs</a>

6. Creating k-armed bandits


<a id="intro"></a>

### Introduction

This is a quick-start guide to using the `webppl-agents` library. For a comprehensive explanation of the ideas behind the library (e.g. MDPs, POMDPs, hyperbolic discounting) and diverse examples of its use, go to the online textbook [agentmodels.org](http://agentmodels.org). 

The webppl-agents library is built around two basic entities: *agents* and *environments*. These entities are combined by *simulating* an agent interacting with a particular environment. The library includes two standard RL environments as examples (Gridworld and Multi-armed Bandits). Four kinds of agent are included. Many combinations of environment and agent are possible. In addition, it's easy to add your own environments and agents -- as we illustrate below. 

Not all environments and agents can be combined. Among environments, we distinguish MDPs (Markov Decision Processes) and POMDPs (Partially Observable Markov Decision Processes). For a POMDP environment, the agent must be a "POMDP agent", which means they maintain a belief distribution on the state[^separation].

[^separation]: This separation of POMDPs and MDPs is not necessary from a theoretical perspective, since POMDPs generalize MDPs. However, the separation is convenient in practice; it allows the MDP code to be short and perspicuous and it provides performance advantages. 

<a id="createMDP"></a>

### Creating your own MDP environment

We begin by creating a very simple MDP environment and running two agents from the library on that environment. 

MDPs are defined [here](http://agentmodels.org/chapters/3a-mdp.html). For use in the library, MDP environments are Javascript objects with the following methods:

>`{transition: ...,  stateToActions: ...}`

The `transition` method is a function from state-action pairs to states (as in the function $$T$$ in the MDP definition). The `stateToAction` method is a mapping from states to the actions that are allowed in that state. (This is often a constant function). 

To run an agent on an MDP, the agent object must have a `utility` method defined on the MDP's state-action space. This method is the agent's "reward" or "utility" function (we use the terms interchangeably). 

#### Creating the Line MDP environment
Our first MDP environment is a discrete line (or one-dimensional gridworld) where the agent can move left or right (starting from the origin). More precisely, the Line MDP is as follows:

- **States:** Points on the integer line (e.g ..., -1, 0, 1, 2, ...).

- **Actions/transitions:** Actions “left”, “right” and “stay” move the agent deterministically along the line in either direction. We represent the actions as $$[-1,0,1]$$ in the code below. 

In our examples, the agent's `startState` is the origin. The utility is 1 at the origin, 3 at the third state right of the origin ("state 3"), and 0 otherwise.

The transition function must also decrement the time. States are objects with a `terminateAfterAction` property. In the example below, `terminateAfterAction` is set to `true` when the state's `timeLeft` attribute gets down to 1; this causes the MDP to terminate. Here is an example state for the Line MDP (it's also the `startState`):

>`{terminateAfterAction: false, timeLeft:5, loc:0}`

NB: The library uses the term "world" in place of "environment".


~~~~
// helper function that decrements time and triggers termination when 
// time elapsed
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

  var stateToActions = function(state){
    return [-1, 0, 1];
  };

  var transition = function(state, action){
    var newLoc = state.loc + action;
    var stateNewLoc = update(state,{loc: newLoc});
    return advanceStateTime(stateNewLoc);
  };

  var world = {
    stateToActions: stateToActions,
    transition: transition
  };

  var startState = {
    timeLeft: totalTime, 
    terminateAfterAction: false, 
    loc: 0
  };

  var utility = function(state, action){    
    var table = { 0: 1, 3: 3 };
    return table[state.loc] ? table[state.loc] : 0;
  };

  return {
    world: world,
    startState: startState,
    utility: utility
  };
};

// save the MDP constructor for use in other codeboxes
wpEditor.put('makeLineMDP', makeLineMDP);
~~~~

To run an agent on this MDP, we use a `makeAgent` constructor and the library function `simulate`. The constructor for MDP agents is `makeMDPAgent`:

>`makeMDPAgent(params, world)`

Agent constructors always have these same two arguments. The `world` argument is required for the agent's internal simulations of possible transitions. The `params` argument specifies the agent's parameters and whether the agent is optimal or biased.

For an optimal agent, the parameters are:

>`{utility: <utility_function>,  alpha: <softmax_alpha>}`

An environment (or "world") and agent are combined with the `simulate` function:

>`simulate(startState, world, agent, outputType)`

Given the utility function defined above, the highest utility state is at location 3 (three steps to the right from the origin). So an optimal agent (who doesn't hyperbolically discount) will move to this location and stay there. 

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

  var world = {
    stateToActions: stateToActions,
    transition: transition
  };

  var startState = {
    timeLeft: totalTime, 
    terminateAfterAction: false, 
    loc: 0
  };

  var utility = function(state, action){    
    var table = {0:1, 3:3};
    return table[state.loc] ? table[state.loc] : 0;
  };

  return {
    world: world,
    startState: startState,
    utility: utility
  };
};
///

// Construct line MDP environment
var totalTime = 5;
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

We described the agent above as "optimal" because it does not hyperbolically discount and it is not myopic. However, we can adjust its "soft-max" noise by modifying the parameter `alpha` and induce sub-optimal behavior. Moreover, we can change the agent's behavior on this MDP by over-writing the utility method in `params`. 

To construct a time-inconsistent, hyperbolically-discounting MDP agent, we include additional attributes in the `params` argument:

>`{ discount:<discount_parameter>, sophisticatedOrNaive: <boolean> }`

These attributes are explained in the [chapter](/chapters/5a-time-inconsistency.html) on hyperbolic discounting. The discounting agent stays at the origin because it isn't willing to "delay gratification" in order to get a larger total reward at location 3.

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

// constructor for the "line" MDP environment:
// argument *totalTime* is the time horizon
var makeLineMDP = function(totalTime){

  var stateToActions = function(state){return [-1, 0, 1];};

  var transition = function(state,action){
    var newLoc = state.loc + action;
    var stateNewLoc = update(state,{loc: newLoc});
    return advanceStateTime(stateNewLoc);
  };

  var world = {
    stateToActions: stateToActions,
    transition: transition
  };

  var startState = {
    timeLeft: totalTime, 
    terminateAfterAction: false, 
    loc: 0};

  var utility = function(state, action){    
    var table = {0:1, 3:3};
    return table[state.loc] ? table[state.loc] : 0;
  };

  return {
    world: world,
    startState: startState,
    utility: utility
  };
};
///

// Construct line MDP environment
var totalTime = 5;
var lineMDP = makeLineMDP(totalTime);
var world = lineMDP.world;

// The lineMDP object also includes a utility function and startState
var utility = lineMDP.utility;
var startState = lineMDP.startState;

// Construct hyperbolic agent
var params = {
  alpha: 1000,
  utility: utility,
  discount: 2,
  sophisticatedOrNaive: 'naive'
};
var agent = makeMDPAgent(params, world);
var trajectory = simulate(startState, world, agent, 'states');
print(trajectory);
~~~~

We've shown how to create your own MDP and then run different agents on that MDP. You can also create your own MDP agent, as we illustrate below.

>**Exercise:** Try some variations of the Line MDP by modifying the `transition` method in the `makeLineMDP` constructor above. For example, change the underlying graph structure from a line into a loop. 

-----------

<a id="gridworld"></a>

### Creating Gridworld MDPs

Gridworld is a standard toy environment for reinforcement learning problems. The library contains a constructor for making a gridworld with your choice of dimensions and reward function. There is also a function for displaying gridworlds in the browser.

We begin by creating a simple gridworld environment (using `makeGridWorld`) and display it using `GridWorld.draw`.

~~~~
// Create a constructor for our gridworld
var makeSimpleGridWorld = function(){

  // '#' indicates a wall, and ' ' indicates a normal cell  
  var ___ = ' ';

  var features = [
    [ ___, ___, ___],
    [ '#', '#', ___],
    [ '#', '#', ___],
    [ ___, ___, ___]
  ];

  // Set the transition noise to zero
  var options = {
    gridFeatures: features,
    transitionNoiseProbability: 0
  };

  return makeGridWorld(options)
};

var simpleGridWorld = makeSimpleGridWorld();
var world = simpleGridWorld.world;

var startState = {
  loc: [0, 0],
  timeLeft: 10,
  terminateAfterAction: false
};

GridWorld.draw(world, {trajectory: [startState]});
~~~~

Gridworld states have a `loc` attribute for the agent's location (using discrete Cartesian coordinates). The agent is able to move up, down, left and right but is not able to stay put.

Having created a gridworld, we construct a utility function (where utility depends only on the agent's grid location) and simulate an optimal MDP agent. 

~~~~
///fold:
// Create a constructor for our gridworld
var makeSimpleGridWorld = function(){

  // '#' indicates a wall, and ' ' indicates a normal cell  
  var ___ = ' ';

  var features = [
    [___, ___, ___],
    ['#', '#', ___],
    ['#', '#', ___],
    [___, ___, ___]
  ];

  // Set the transition noise to zero
  var options = {
    gridFeatures: features,
    transitionNoiseProbability: 0
  };

  return makeGridWorld(options)
};

var simpleGridWorld = makeSimpleGridWorld();
var world = simpleGridWorld.world;

var startState = {loc: [0,0],
    timeLeft: 10,
    terminateAfterAction: false};

///

// `isEqual` is in *underscore* (included in webppl-agents)
var utility = function(state, action){
  return _.isEqual(state.loc, [0, 3]) ? 1 : 0
};

var params = { utility: utility, alpha: 1000 };
var agent = makeMDPAgent(params, world);
var trajectory = simulate(startState, world, agent);
GridWorld.draw(world, {trajectory: trajectory});
~~~~

You can create terminal gridworld states by using features with a name. These named-features can also be used to create a utility function without specifying grid coordinates. 

~~~~
var makeSimpleGridWorld = function(){

  // '#' indicates a wall, and ' ' indicates a normal cell  
  var ___ = ' ';

  // named features are terminal
  var G = { name: 'gold' };
  var S = { name: 'silver' };
  
  var features = [
    [ G , ___, ___],
    [ S , ___, ___],
    ['#', '#', ___],
    ['#', '#', ___],
    [___, ___, ___]
  ];

  // Set the transition noise to zero
  var options = {
    gridFeatures: features,
    transitionNoiseProbability: 0
  };
  
  return makeGridWorld(options)
};

var simpleGridWorld = makeSimpleGridWorld();
var world = simpleGridWorld.world;

var startState = {
  loc: [0, 0],
  timeLeft: 10,
  terminateAfterAction: false
};

// The *makeUtility* method allows you to define
// a utility function in terms of named features
var makeUtility = simpleGridWorld.makeUtility;
var table = {
  gold: 2, 
  silver: 1.8, 
  timeCost: -0.5
};
var utility = makeUtility(table)

var params = { utility: utility, alpha: 1000 };
var agent = makeMDPAgent(params, world);
var trajectory = simulate(startState, world, agent);
GridWorld.draw(world, { trajectory: trajectory });
~~~~

There are many examples using gridworld in agentmodels.org, starting from this [chapter](/chapters/3b-mdp-gridworld.html).


-------

<a id="agents"></a>

### Creating your own agents
As well as creating your own environments, it is straightfoward to create your own agents for MDPs and POMDPs. Much of agentmodels.org is a tutorial on creating agents (e.g. optimal agents, myopic agents, etc.). Rather than recapitulate agentmodels.org, this section is brief and focuses on the basic interface that agents need to present.

We begin by creating an agent that chooses actions uniformly at random. To run on agent on an environment using the `simulate` function, an agent object must have an `act` method and a `params` attribute. The `act` method is a function from states to a distribution on the available actions. The `params` attribute indicates whether or not the agent is an MDP or POMDP agent.

We use the simple gridworld environment from the codebox above. 

~~~~
// Build gridworld environment
///fold:
var makeSimpleGridWorld = function(){

  // '#' indicates a wall, and ' ' indicates a normal cell  
  var ___ = ' ';

  // named features are terminal
  var G = { name: 'gold' };
  var S = { name: 'silver' };
  
  var features = [
    [ G , ___, ___],
    [ S , ___, ___],
    ['#', '#', ___],
    ['#', '#', ___],
    [___, ___, ___]
  ];

  // Set the transition noise to zero
  var options = {
    gridFeatures: features,
    transitionNoiseProbability: 0
  };
  
  return makeGridWorld(options)
};

var simpleGridWorld = makeSimpleGridWorld();
var world = simpleGridWorld.world;

var startState = {
  loc: [0, 0],
  timeLeft: 10,
  terminateAfterAction: false
};

// The *makeUtility* method allows you to define
// a utility function in terms of named features
var makeUtility = simpleGridWorld.makeUtility;
var table = {
  gold: 2, 
  silver: 1.8, 
  timeCost: -0.5
};
var utility = makeUtility(table)
///

var actions = ['u', 'd', 'l', 'r'];

var act = function(state){
  return Infer(
    { method: 'enumerate' }, 
    function(){ return uniformDraw(actions); })
};

// Since params has no *POMDP* attribute, the agent
// defaults to being an MDP agent
var randomAgent = { act: act, params: {} };
var trajectory = simulate(startState, world, randomAgent);
GridWorld.draw(world, { trajectory: trajectory });
~~~~

In gridworld the same actions are available in each state. When the actions available depend on the state, the agent's `act` function needs access to the environment's `stateToActions` method.

~~~~
///fold:
// Create a constructor for our gridworld
var makeSimpleGridWorld = function(){

  // '#' indicates a wall, and ' ' indicates a normal cell  
  var ___ = ' ';

  // named features are terminal
  var G = { name: 'gold' };
  var S = { name: 'silver' };
  
  var features = [
    [ G , ___, ___],
    [ S , ___, ___],
    ['#', '#', ___],
    ['#', '#', ___],
    [___, ___, ___]
  ];

  // Set the transition noise to zero
  var options = {
    gridFeatures: features,
    transitionNoiseProbability: 0
  };
  
  return makeGridWorld(options)
};

var simpleGridWorld = makeSimpleGridWorld();
var world = simpleGridWorld.world;

var startState = {
  loc: [0, 0],
  timeLeft: 10,
  terminateAfterAction: false
};

// The *makeUtility* method allows you to define
// a utility function in terms of named features
var makeUtility = simpleGridWorld.makeUtility;
var table = {
  gold: 2, 
  silver: 1.8, 
  timeCost: -0.5
};
var utility = makeUtility(table)
///

 
var makeRandomAgent = function(world){
  var stateToActions = world.stateToActions;
  
  var act = function(state){
    return Infer(
      { method: 'enumerate' }, 
      function(){return uniformDraw(stateToActions(state));})
  };
 
  return { act: act, params: {} };
};

var randomAgent = makeRandomAgent(world);
var trajectory = simulate(startState, world, randomAgent);

GridWorld.draw(world, { trajectory: trajectory });
~~~~

In the example above, the agent constructor `makeRandomAgent` takes the environment (`world`) as an argument in order to access `stateToActions`. Agent constructors will typically also use the environment's `transition` method to internally simulate state transitions.

>**Exercise:** Implement an agent who takes the action with highest expected utility under the random policy. (You can do this by making use of the codebox above. Use the `makeRandomAgent` and `simulate` function within a new agent constructor.)

In addition to writing agents from scratch, you can build on the agents available in the library. 

>**Exercise:** Start with the optimal MDP agent found [here](https://github.com/agentmodels/webppl-agents/blob/master/src/agents/makeMDPAgent.wppl#L3). Create a variant of this optimal agent that takes "epsilon-greedy" random actions instead of softmax random actions. 

--------

<a id="createPOMDP"></a>

### Creating POMDPs

POMDPs are introduced in agentmodels.org in this [chapter](/chapters/3c-pomdp.html). This section explains how to create your own POMDPs for use in the library.

As we explained above, MDPs in webppl-agents are objects with a `transition` method and a `stateToActions` method. POMDPs also have a `transition` method. Instead of `stateToActions`, they have a `beliefToActions` method, which maps a belief distribution over states to a set of available actions. POMDPs also have an `observe` method, which maps states to observations (typically represented as strings).

Here is a simple POMDP based on the "Line MDP" example above. The agent moves along the integer line as before. This time the agent is uncertain whether or not there is high reward at location 3. The agent can only find out by moving to location 3 and receiving an observation.

~~~~

// States have the same structure as in MDPs:
// the transition method needs to decrement
// the state's *timeLeft* attribute until termination

var advanceStateTime = function(state){
  var newTimeLeft = state.timeLeft - 1;
  return update(state, { 
    timeLeft: newTimeLeft,
    terminateAfterAction: newTimeLeft > 1 ? state.terminateAfterAction : true
  });
};


var makeLinePOMDP = function(){

  var beliefToActions = function(belief){return [-1, 0, 1];};
  
  var transition = function(state, action){
    var newLoc = state.loc + action;
    var stateNewLoc = update(state,{loc: newLoc});
    return advanceStateTime(stateNewLoc);
  };
  
  var observe = function(state){
    if (state.loc == 3){
      return state.treasureAt3 ? 'treasure' : 'no treasure';
    }
    return 'noObservation';
  };

  return {beliefToActions:beliefToActions, 
          transition:transition, 
          observe:observe};  
          
};
~~~~

To simulate an agent on this POMDP, we need to create a "POMDP" agent. POMDP agents have an `act` method which maps *beliefs* (rather than *states*) to distributions on actions. They also have an `updateBelief` method, mapping beliefs and observations to an updated belief. 

This example uses the optimal POMDP agent. To construct a POMDP agent, we need to specify the agent's starting belief distribution on states. Here we assume the agent has a uniform distribution on whether or not there is "treasure" at location 3.

~~~~
///fold:
var advanceStateTime = function(state){
  var newTimeLeft = state.timeLeft - 1;
  return update(state, { 
    timeLeft: newTimeLeft,
    terminateAfterAction: newTimeLeft > 1 ? state.terminateAfterAction : true
  });
};

var makeLinePOMDP = function(){

  var beliefToActions = function(belief){return [-1, 0, 1];};
  
  var transition = function(state, action){
    var newLoc = state.loc + action;
    var stateNewLoc = update(state,{loc: newLoc});
    return advanceStateTime(stateNewLoc);
  };
  
  var observe = function(state){
    if (state.loc == 3){
      return state.treasureAt3 ? 'treasure' : 'no treasure';
    }
    return 'noObservation';
  };

  return {beliefToActions:beliefToActions, 
          transition:transition, 
          observe:observe};  
};
///  

var utility = function(state, action){    
  if (state.loc==3 && state.treasureAt3){return 5;}
  if (state.loc==0){return 1;}
  return 0;
};

var trueStartState = {timeLeft: 7, 
                      terminateAfterAction: false, 
                      loc: 0,
                      treasureAt3: false};

var alternativeStartState = update(trueStartState, {treasureAt3: true});
var possibleStates = [trueStartState, alternativeStartState];

var priorBelief = Categorical({ps: [.5, .5], vs: possibleStates});

var params = {alpha:1000,              
              utility:utility, 
              priorBelief: priorBelief,  
              optimal: true
             };

var world = makeLinePOMDP();
var agent = makePOMDPAgent(params, world);
var trajectory = simulate(trueStartState, world, agent, 'states');
print(trajectory)
~~~~

In POMDPs the agent does not directly observe their current state. However, in the Line POMDP (above) the "location" part of the agent's state is always known by the agent. The part of the state that is unknown is whether `treasureAt3` is true. So we could factor the state into attributes that are always known ("manifest") and parts that are not ("latent"). This factoring of the state can speed up the POMDP agent's belief-updating and is used for the POMDP environments in the library. The following codebox shows a factored version of the Line POMDP:

~~~~
///fold:
var advanceStateTime = function(state){
  var newTimeLeft = state.timeLeft - 1;
  return update(state, { 
    timeLeft: newTimeLeft,
    terminateAfterAction: newTimeLeft > 1 ? state.terminateAfterAction : true
  });
};
///

var makeLinePOMDP = function(){
  var manifestStateToActions = function(manifestState){return [-1, 0, 1];};
  
  var transition = function(state, action){
    var newLoc = state.manifestState.loc + action;
    var manifestStateNewLoc = update(state.manifestState,{loc: newLoc});
    var newManifestState = advanceStateTime(manifestStateNewLoc);
    return {manifestState: newManifestState, latentState: state.latentState};
  };
  
  var observe = function(state){
    if (state.manifestState.loc == 3){
      return state.latentState.treasureAt3 ? 'treasure' : 'no treasure';
    }
    return 'noObservation';
  };
  
  return {manifestStateToActions:manifestStateToActions, 
          transition:transition, 
          observe:observe};
};


var utility = function(state, action){    
  if (state.manifestState.loc==3 && state.latentState.treasureAt3){return 5;}
  if (state.manifestState.loc==0){return 1;}
  return 0;
};

var trueStartState = {manifestState: {timeLeft: 7, 
                                      terminateAfterAction: false, 
                                      loc: 0},
                      latentState: {treasureAt3: false}
                     };

var alternativeStartState = update(trueStartState, 
                                   {latentState: {treasureAt3: true}});
var possibleStates = [trueStartState, alternativeStartState];

var priorBelief = Categorical({ps: [.5, .5], vs: possibleStates});

var params = {alpha:1000,              
              utility:utility, 
              priorBelief: priorBelief,  
              optimal: true
             };

var world = makeLinePOMDP();
var agent = makePOMDPAgent(params, world);  
var trajectory = simulate(trueStartState, world, agent, 'states');
print(trajectory)
~~~~



---------


### Footnotes



