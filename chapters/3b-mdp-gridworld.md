---
layout: chapter
title: "MDPs and Gridworld in WebPPL"
description: We extend the previous setup to stochastic actions (softmax) and transitions, and introduce policies and expected action values.
---

## Introduction

We introduced Markov Decision Processes (MDPs) with the example of Bob moving around a city with the aim of efficiently finding a good restaurant. We return to this example in later chapters, considering how agents with *uncertainty* or *bias* will behave in this environment. This chapter explores some key features of MDPs that will be important throughout the tutorial.

### Hiking in Gridworld

We begin by introducing a new gridworld MDP:

> **Hiking Problem**:
>Suppose that Alice is hiking. There are two peaks nearby, denoted "West" and "East". The peaks provide different views and Alice must choose between them. South of Alice's starting position is a steep hill. Falling down the hill would result in painful (but non-fatal) injury and end the hike early.

We represent Alice's hiking problem with a Gridworld similar to Bob's Restaurant Choice example. The peaks are terminal states, providing differing utilities. The steep hill is represented by a row of terminal state, each with identical negative utility. Each timestep before Alice reaches a terminal state incurs a "time cost", which is negative to represent the fact that Alice prefers a shorter hike. 

~~~~
// draw_hike
var world = makeHike(0);
var startState = {loc: [0, 1]};
GridWorld.draw(world, {trajectory: [startState]});
~~~~

We start with a *deterministic* transition function. In this case, Alice's risk of falling down the steep hill is solely due to softmax noise in her action choice (which is minimal in this case). The agent model is the same as the one at the end of [Chapter III.1](/chapters/3a-mdp.html'). We wrap the functions `agent`, `expectedUtility` and `simulate` in a function `mdpSimulateGridworld`. The following codebox defines this function and we use it later on without defining it (since it's in the WebPPL Gridworld library). 

~~~~
var makeMDPAgent = function(params, world) {
  var stateToActions = world.stateToActions;
  var transition = world.transition;
  var utility = params.utility;
  var alpha = params.alpha;

  var act = dp.cache( 
    function(state){
      return Enumerate(function(){
        var action = uniformDraw(stateToActions(state));
        var eu = expectedUtility(state, action);
        factor(alpha * eu);
        return action;
      });
    });
  
  var expectedUtility = dp.cache(
    function(state, action){
      var u = utility(state, action);
      if (state.terminateAfterAction){
        return u; 
      } else {
        return u + expectation( Enumerate(function(){
          var nextState = transition(state, action); 
          var nextAction = sample(act(nextState));
          return expectedUtility(nextState, nextAction);
        }));
      }
    });
  
  return {
    params : params,
    expectedUtility : expectedUtility,
    act: act
  };
};

var simulateMDP = function(startState, world, agent) {
  var act = agent.act;
  var expectedUtility = agent.expectedUtility;
  var transition = world.transition;

  var sampleSequence = function(state) {
    var action = sample(act(state));
    var nextState = transition(state, action);
    var out = [state, action];
    return state.terminateAfterAction ? [out]
      : [out].concat(sampleSequence(nextState));
  };
  return sampleSequence(startState);
};

var mdpTableToUtilityFunction = function(table, feature) {
  return function(state, action) {
    var stateFeatureName = feature(state).name;
    
    return stateFeatureName ? table[stateFeatureName] : table.timeCost;
  };
};

// parameters for world
var transitionNoiseProb = 0;

var world = makeHike(transitionNoiseProb);
var feature = world.feature;

var startState = {loc: [0,1],
		          timeLeft: 10,
				  terminateAfterAction: false,
				  timeAtRestaurant: 1};

// parameters for agent
var utilityTable = {East: 10, West: 1, Hill: -10, timeCost: -.1};
var utility = mdpTableToUtilityFunction(utilityTable, feature);
var alpha = 100;
var agent = makeMDPAgent({utility: utility, alpha: alpha}, world);

var trajectory = simulateMDP(startState, world, agent);

// GridWorld.draw(world, {trajectory: trajectory});
var displayTrajectory = function(trajectory) {
  var stateActionToLocAction = function(stateAction) {
    return [stateAction[0].loc, stateAction[1]];
  };
  return map(stateActionToLocAction, trajectory);
};

displayTrajectory(trajectory)
~~~~

## Hiking under the influence 

TODO: change this example?

If we set the softmax noise parameter `alpha=10`, the agent will often make sub-optimal decisions. While not realistic in Alice's situation, this might describe a confused or intoxicated agent. Since the agent is noisy, we sample many trajectories to approximate the agent's distribution on trajectories using the built-in function `Rejection`. The main use for `Rejection` is inference by rejection sampling. However, here we use `Rejection` without any `condition` or `factor` statement simply to summarize the agent's noisy behavior. We do this by computing the *length* of the agent's trajectories, since suboptimal actions will lead to less efficient routes to the East peak. (Note that, if the agent is left of a wall and takes the action "left", then the agent doesn't move anywhere. If the time cost isn't high, the noisy agent will often move towards the walls).

~~~~
// Parameters for building Hiking MDP
var utilityTable = { east: 10, west: 1, hill: -10, timeCost: -.1 };
var startState = [0, 1];

var alpha = 10;
var transitionNoiseProb = 0;
var params = makeHike(transitionNoiseProb, alpha, utilityTable);

var totalTime = 12;

// Approximate distribution on trajectories using 500 samples
var numRejectionSamples = 500;
var trajectoryDist = mdpSimulateGridworld(
  startState, totalTime, params, numRejectionSamples);

// Show distribution on length of trajectories
var trajectoryLengthDist = Enumerate(function(){
  return sample(trajectoryDist).length;
});
viz.print(trajectoryLengthDist);

// Show a random trajectory
var trajectory = sample(trajectoryDist);
GridWorld.draw(params, {labels: params.labels, trajectory : trajectory});
~~~~

### Exercise

Sample some of the noisy agent's trajectories by repeatedly clicking "run". Does the agent ever fall down the hill? Why not? By varying the softmax noise `alpha` and other parameters, find a setting where the agent's modal trajectory length is the same as the `totalTime`. (Don't modify the `transitionNoiseProb` parameter for this exercise). 
<!-- let alpha=0.5 and action cost = -.01 -->


## Hiking with stochastic transitions

When softmax noise is high, the agent will make many small "mistakes" (i.e. suboptimal actions given the agent's own preferences), but few large mistakes. In contrast, sources of noise in the environment will change the agent's state transitions independent of the agent's preferences. In the hiking example, imagine that the weather is very wet and windy. As a result, Alice will sometimes intend to go one way but actually go another way (because she slips in the mud). In this case, the shorter route to the peaks might be too risky for Alice.

To model bad weather, we assume that at every timestep, there is a constant independent probability `transitionNoiseProb` of the agent moving orthogonally to their intended direction. The independence assumption is unrealistic (if a location is slippery at one timestep it is more likely slippery the next), but it is simple and satisfies the Markov assumption.

Setting `transitionNoiseProb=0.1`, the agent's first intended action is "up" instead of "right", because the shorter route is risky. 

### Exercise

Keeping `transitionNoiseProb=0.1`, find settings for the arguments to `makeHike` such that the agent goes "right" instead of "up". <!-- put up timeCost to -.5 or so --> 

~~~~
// Parameters for building Hiking MDP
var transitionNoiseProb = 0.1;
var world = makeHike(transitionNoiseProb);
var feature = world.feature;

var utilityTable = { East: 10, West: 1, Hill: -10, timeCost: -.1 };
var utility = mdpTableToUtilityFunction(utilityTable, feature);
var alpha = 100;
var agent = makeMDPAgent({utility: utility, alpha: alpha}, world)

var startState = {loc: [0,1],
                  timeLeft: 12,
				  terminateAfterAction: false};

var trajectory = simulateMDP(startState, world, agent);
// draw trajectory


var displayTrajectory = function(trajectory) {
  var stateActionToLocAction = function(stateAction) {
    return [stateAction[0].loc, stateAction[1]];
  };
  return map(stateActionToLocAction, trajectory);
};

displayTrajectory(trajectory)
~~~~

In a world with stochastic transitions, the agent sometimes finds itself in a state it did not intend to reach. The functions `agent` and `expectedUtility` (inside `mdpSimulateGridworld`) implicitly compute the expected utility of actions for every possible future state, including states that the agent will try to avoid. In the MDP literature, this function from states and remaining time to actions (or distributions on actions) is called a *policy*. (For infinite-horizon MDPs, policies are simply functions from states to actions.)

The example above shows that the agent chooses the long route, steering clear of the steep hill. At the same time, the agent computes what to do if it ends up moving right on the first action. (The code below doesn't prove this; it just illustrates what the agent would do if it moved right.)

~~~~
// Parameters for building Hiking MDP
var transitionNoiseProb = 0.1;
var world = makeHike(transitionNoiseProb);
var feature = world.feature;

var utilityTable = { East: 10, West: 1, Hill: -10, timeCost: -.1 };
var utility = mdpTableToUtilityFunction(utilityTable, feature);
var alpha = 100;
var agent = makeMDPAgent({utility: utility, alpha: alpha}, world);

// Change start state from [1,0] to [1,1] and reduce time
var startState = {loc: [1,1],
                  timeLeft : 11,
				  terminateAfterAction: false,
				  timeAtRestaurant: 1};

var trajectory = simulateMDP(startState, world, agent);

// draw trajectory

var displayTrajectory = function(trajectory) {
  var stateActionToLocAction = function(stateAction) {
    return [stateAction[0].loc, stateAction[1]];
  };
  return map(stateActionToLocAction, trajectory);
};

displayTrajectory(trajectory)
~~~~

Extending this idea, we can return and visualize the expected values of actions that the agent *could have taken* during their trajectory. For each state in a trajectory, we compute the expected value of each possible action (given the state and remaining time). The resulting numbers are analogous to Q-values in infinite-horizon MDPs. 

The expected values we seek to display are already being computed: we add a function addition to `mdpSimulateGridworld` in order to return them.

~~~~
// trajectory must consist only of states. This can be done by calling
// *simulateMDP* with an additional final argument 'states'.
var getExpectedUtilitiesMDP = function(stateTrajectory, world, agent) {
  var eu = agent.expectedUtility;
  var stateToActions = world.stateToActions;
  var getAllExpectedUtilities = function(state) {
    var availableActions = stateToActions(state);
    return [state, map(function(action){return eu(state, action);},
		       availableActions)];
  };
  return map(getAllExpectedUtilities, stateTrajectory);
};

// long route better and takes long route

var noiseProb = 0.03;
var world = makeHike(noiseProb, {big: true});
var feature = world.feature;

var alpha = 100;
var utilityTable = {East: 10, West: 7, Hill : -40, timeCost: -0.4};
var utility = mdpTableToUtilityFunction(utilityTable, feature);
var agent = makeMDPAgent({utility: utility, alpha: alpha}, world);

var startState = {loc: [1,1],
		  timeLeft: 12,
		  terminateAfterAction: false,
		  timeAtRestaurant: 1};

var trajectory = simulateMDP(startState, world, agent, 'states');
var locs1 = map(function(state){return state.loc;}, trajectory);
var eus = getExpectedUtilitiesMDP(trajectory, world, agent);
// figure out nice way to display locations and expected utilities


// GridWorld.draw(params, { 
//   labels: params.labels,
//   trajectory: trajectoryExpUtilities, 
//   expUtilities: trajectoryExpUtilities
// });


// TODO FIX: stochastic expUtilities and doesnt take highest EU
// action despite low noise.

// note that this problem persists with the new gridworld mdp functions

var noiseProb = .04;
var world = makeHike(noiseProb, {big: true});
var feature = world.feature;

var alpha = 100;
var utilityTable = { East: 15, West: 7, Hill: -40, timeCost: -.8 };
var utility = mdpTableToUtilityFunction(utilityTable, feature);
var agent = makeMDPAgent({utility: utility, alpha: alpha}, world);

var startState = {loc: [1,1],
		          timeLeft: 14,
				  terminateAfterAction: false,
				  timeAtRestaurant: 1};

var trajectory = simulateMDP(startState, world, agent, 'states');
var eus = getExpectedUtilitiesMDP(trajectory, world, agent);

var locs2 = map(function(state){return state.loc;}, trajectory);
// locs1;
locs2;
~~~~


--------------

[Table of Contents](/)
