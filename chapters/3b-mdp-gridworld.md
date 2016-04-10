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
// define_agent_simulate

var makeMDPAgent = function(params, world) {  
  var stateToActions = world.stateToActions;
  var transition = world.transition;
  var utility = params.utility;

  var act = dp.cache( 
    function(state){
      return Enumerate(function(){
        var action = uniformDraw(stateToActions(state));
        var eu = expectedUtility(state, action);
        factor(params.alpha * eu);
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

var simulateMDP = function(startState, world, agent, outputType) {
  // if outputType is undefined, default to stateAction
  var act = agent.act;
  var transition = world.transition;

  var selectOutput = function(state, action) {
    var table = {states: state,
		         actions: action,
				 stateAction: [state, action]};
    return outputType ? table[outputType] : table.stateAction;
  };

  var sampleSequence = function(state) {
    var action = sample(act(state));
    var nextState = transition(state, action);
    var out = selectOutput(state, action);
    return state.terminateAfterAction ? [out]
      : [out].concat(sampleSequence(nextState));
  };
  return sampleSequence(startState);
};


// parameters for world
var transitionNoiseProb = 0;
var world = makeHike(transitionNoiseProb);
var startState = {loc: [0,1],
		          timeLeft: 12,
				  terminateAfterAction: false};

// parameters for agent
var utilityTable = {East: 10, West: 1, Hill: -10, timeCost: -.1};
var utility = makeHikeUtilityFunction(world, utilityTable);
var agent = makeMDPAgent({utility: utility, alpha: 1000}, world);
var trajectory = simulateMDP(startState, world, agent, 'states');


GridWorld.draw(world, {trajectory: trajectory});
~~~~

>**Exercise**: Adjust the parameters `utilityTable` in order to produce the following behaviors:

>1. The agent does directly to "West".
>2. The agent takes the long way around to "West".
>3. The agent sometimes goes to the Hill at $$[1,0]$$. The probability of this outcome is close to the most likely trajectory for the agent. 
<!-- 3 is obtained by making timeCost positive and Hill better than alternatives -->


### Hiking with stochastic transitions

Imagine that the weather is very wet and windy. As a result, Alice will sometimes intend to go one way but actually go another way (because she slips in the mud). In this case, the shorter route to the peaks might be too risky for Alice.

To model bad weather, we assume that at every timestep, there is a constant independent probability `transitionNoiseProb` of the agent moving orthogonally to their intended direction. The independence assumption is unrealistic (if a location is slippery at one timestep it is more likely slippery the next), but it is simple and satisfies the Markov assumption for MDPs.

Setting `transitionNoiseProb=0.1`, the agent's first action is "up" instead of "right", because the shorter route is now risky. 

~~~~
// parameters for world
var transitionNoiseProb = 0.1;
var world = makeHike(transitionNoiseProb);
var startState = {loc: [0,1],
		          timeLeft: 12,
				  terminateAfterAction: false};

// parameters for agent
var utilityTable = { East: 10, West: 1, Hill: -10, timeCost: -.1 };
var utility = makeHikeUtilityFunction(world, utilityTable);
var agent = makeMDPAgent({utility: utility, alpha: 1000}, world);
var trajectory = simulateMDP(startState, world, agent, 'states');

GridWorld.draw(world, {trajectory: trajectory});
~~~~

>**Exercise:**

>1. Keeping `transitionNoiseProb=0.1`, find settings for `utilityTable` such that the agent goes "right" instead of "up".
>2. Set `transitionNoiseProb=0.01`. Change a single parameter in `utilityTable` such that the agent goes "right" (there are multiple ways to do this). 
<!-- put up timeCost to -1 or so --> 

### Noisy transitions vs. Noisy agents
It's important to distinguish noise in the transition function from the softmax noise for the agent's selection of actions. Noise in the transition function is a representation of randomness in the world. This is easiest to think about in casino games and other games of chance [^noise]. In playing a casino game (without cheating) any agent will need to model the randomness in the game. By contrast, softmax noise is a property of an agent. For example, we can vary the behavior otherwise identical agents by varying their parameter $$\alpha$$.

Unlike transition noise, softmax noise will have less influence on the agent's planning for the Hiking Problem. Since it's so bad to fall down the hill, the softmax agent will very rarely do so even if they take the short route. The softmax agent is like a lazy person who takes mildy inefficient routes but "pulls himself together" when the stakes are high.

[^noise]: An agent might also use a simplified representation of the world that treats a complex set of deterministic rules as random. In this sense, agents will vary in whether they represent an MDP as stochastic or not. We won't consider that case in this tutorial. 

>**Exercise:** Use the codebox below to explore different levels of softmax noise. Find a setting of `utilityTable` and `alpha` such that the agent goes to West and East equally often and nearly always takes the most direct route to both East and West. Included below is code for simulating many trajectories and return the trajectory length. You can extend this code to measure whether the route taken by the agent is direct or not. (Note that while the softmax agent here is able to "backtrack" or return to its previous location, in later Gridworld examples we rule out backtracking as a possible action).  

~~~~
// parameters for world
var transitionNoiseProb = 0.1;
var world = makeHike(transitionNoiseProb);
var startState = {loc: [0,1],
		          timeLeft: 12,
				  terminateAfterAction: false};

// parameters for agent
var alpha = 1;
var utilityTable = { East: 10, West: 1, Hill: -10, timeCost: -.1 };
var utility = makeHikeUtilityFunction(world, utilityTable);
var agent = makeMDPAgent({utility: utility, alpha: alpha}, world);

// generate a single trajectory
var trajectory = simulateMDP(startState, world, agent, 'states');
GridWorld.draw(world, {trajectory: trajectory});

// run 100 iid samples of the function *lengthTrajectory*
var lengthTrajectory = function(){return simulateMDP(startState, world, agent).length;};
var trajectoryERP = Rejection( lengthTrajectory, 100);
viz.vegaPrint(trajectoryERP);

~~~~


### Stochastic transitions: plans and policies

We return to the case of a stochastic environment with very low softmax action noise. In a stochastic environment, the agent sometimes finds themself in a state they did not intend to reach. The functions `agent` and `expectedUtility` (inside `makeMDPAgent`) implicitly compute the expected utility of actions for every possible future state, including states that the agent will try to avoid. In the MDP literature, this function from states and remaining time to actions (or distributions on actions) is called a *policy*. (For infinite-horizon MDPs, policies are simply functions from states to actions.) Since policies take into account every possible contingency, they are quite different from the everyday notion of a "plan". 

Consider the example from above where the agent takes the long route because of the risk of falling down the hill. If we generate a single trajectory for the agent, they will likely take the long route. However, if we generated many trajectory, we would sometimes see the agent move "right" instead of "up" on their first move. Before taking this first action, the agent implicitly computes what to do if they end up moving right. The next codebox illustrates what they'd do: 

~~~~
// policy

// parameters for world
var transitionNoiseProb = 0.1;
var world = makeHike(transitionNoiseProb);

// change start from [0,1] to [1,1] and timeLeft to 11
var startState = {loc: [1,1],
		          timeLeft: 12-1,
				  terminateAfterAction: false};

// parameters for agent
var utilityTable = { East: 10, West: 1, Hill: -10, timeCost: -.1 };
var utility = makeHikeUtilityFunction(world, utilityTable);
var agent = makeMDPAgent({utility: utility, alpha: 1000}, world);
var trajectory = simulateMDP(startState, world, agent, 'states');

GridWorld.draw(world, {trajectory: trajectory});
~~~~

Extending this idea, we can return and visualize the expected values of each action the agent *could have taken* during their trajectory. For each state in a trajectory, we compute the expected value of each possible action (given the state and remaining time). The resulting numbers are analogous to Q-values in infinite-horizon MDPs. 

The expected values we seek to display are already being computed implicitly: we add a function addition to `getExpectedUtilitiesMDP` in order to return them. We then plot these on the Gridworld itself. The numbers in each grid cell are the expected utilities of moving in the corresponding directions. This allows us to see how close the agent was to taking the short route as opposed to the long route. (Also note that if the difference in expected utility between two actions is small then a noisy agent will take each of them with nearly equal probability). 

TODO: fix this codebox. There is an example in John's examples/hyperbolic/generative_examples.wppl (which uses src/hyperbolic.wppl). It should be possible to port over that code. 

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
		          terminateAfterAction: false
		          // COMMENT OUT FOR TESTING timeAtRestaurant: 1
			     };

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
