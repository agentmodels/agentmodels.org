---
layout: chapter
title: "Sequential decision problems, part II: Gridworld"
description: Hiking example, softmax agent, stochastic transitions, policies, expected values of possible actions, discounting
---

## MDP Example: Hiking

We introduced Markov Decision Processes (MDPs) with the example of Bob moving around a city with the aim of efficiently finding a good restaurant. We return to this example in later chapters, considering how agents with *uncertainty* or *bias* will behave in this environment. This chapter explores some key features of MDPs that will be important throughout the tutorial.

We begin by introducing a new gridworld MDP:

### Hiking MDP:
Suppose that Alice is hiking. There are two peaks nearby, denoted "West" and "East". The peaks provide different views and Alice must choose between them. South of Alice's starting position is a steep hill. Falling down the hill would result in painful (but non-fatal) injury and end the hike early.

We represent Alice's hiking problem with a gridworld similar to Bob's Restaurant Choice example. The peaks are terminal states, providing differing utilities. The steep hill is represented by a row of terminal state, each with identical negative utility. Each timestep before Alice reaches a terminal state incurs a "time-cost", which is negative. This means Alice prefers a shorter hike. 

~~~~
var utilityTable = {east:10, west:1, hill:-10, timeCost:-.1};
var alpha = 100;
var transitionNoiseProb = 0;

var params = makeHike(transitionNoiseProb, alpha, utilityTable);

var startState = [0,1];
GridWorld.draw(params, {labels: params.labels, trajectory: [[startState]]});
~~~~

We start with a *deterministic* transition function. This means that Alice's only risk of falling down the steep hill is due to softmax noise in her actions (which is minimal in this case). The agent model is the same as the end of [Chapter III.1](/chapters/3a-mdp.html'). We wrap the functions `agent`, `expUtility` and `simulate` in a function `mdpSimulateGridworld`. The following code box defines this function and we use it later on without defining it (since it is also included in the WebPPL Gridworld library). 

~~~~
var mdpSimulateGridworld = function(startState, totalTime, params, numRejectionSamples){
  var transition = params.transition;
  var utility = params.utility;
  var actions = params.actions;
  var isTerminal = function(state){return state[0]=='dead';};


  var agent = dp.cache(function(state, timeLeft){
    return Enumerate(function(){
      var action = uniformDraw(actions);
      var eu = expUtility(state, action, timeLeft);    
      factor( params.alpha * eu);
    return action;
    });      
  });
  
  
  var expUtility = dp.cache(function(state, action, timeLeft){
    var u = utility(state,action);
    var newTimeLeft = timeLeft - 1;
    
    if (newTimeLeft == 0 | isTerminal(state)){
      return u; 
    } else {                     
      return u + expectation( Enumerate(function(){
        var nextState = transition(state, action); 
        var nextAction = sample(agent(nextState, newTimeLeft));
        return expUtility(nextState, nextAction, newTimeLeft);  
      }));
    }                      
  });
  
  var simulate = function(startState, totalTime){
  
    var sampleSequence = function(state, timeLeft){
      if (timeLeft == 0 | isTerminal(state)){
        return [];
      } else {
      var action = sample(agent(state, timeLeft));
        var nextState = transition(state,action); 
        return [[state,action]].concat( sampleSequence(nextState,timeLeft-1 ))
      }
      };
    // repeatedly sample trajectories for the agent and return ERP
    return Rejection(function(){return sampleSequence(startState, totalTime);}, numRejectionSamples);
  };

  return simulate(startState, totalTime);
};

// parameters for building Hiking MDP
var utilityTable = {east:10, west:1, hill:-10, timeCost:-.1};
var startState = [0,1];

// parameters for noisy agent (but no transition noise)
var alpha = 100;
var transitionNoiseProb = 0;
var params = makeHike(transitionNoiseProb, alpha, utilityTable);

var totalTime = 12;
var numRejectionSamples = 1;
var out = sample( mdpSimulateGridworld(startState, totalTime, params, 
    numRejectionSamples) );
GridWorld.draw(params, {labels: params.labels, trajectory : out});

~~~~

## Hiking under the influence 
If we set the softmax noise parameter `alpha=10`, the agent will often make sub-optimal decisions. While not realistic in Alice's situation, this might describe a confused or intoxicated agent. Since the agent is noisy, we sample many trajectories to approximate the agent's distribution on trajectories.

To construct an ERP based on these samples we use the built-in function `Rejection`. The main use for `Rejection` is inference by rejection sampling. However, here we use `Rejection` without any `condition` or `factor` statement simply to summarize the agent's noisy behavior. We do this by computing the *length* of the agent's trajectories -- since suboptimal actions will lead to less efficient routes to the East peak. (Note that if the agent is left of a wall and takes the action "left", then the agent doesn't move anywhere. In this environment, the noisy agent will often move towards the walls). 

~~~~
// parameters for building Hiking MDP
var utilityTable = {east:10, west:1, hill:-10, timeCost:-.1};
var startState = [0,1];

var alpha = 10;
var transitionNoiseProb = 0;
var params = makeHike(transitionNoiseProb, alpha, utilityTable);

var totalTime = 12;

// Compute lengths of 500 trajectory
var numRejectionSamples = 500;
var erp = Enumerate( function(){
  return sample( 
    mdpSimulateGridworld(startState, totalTime, params, numRejectionSamples)).length;
});
viz.print(erp);

// Display a random trajectory
var trajectory = sample(mdpSimulateGridworld(startState, totalTime, params, numRejectionSamples));
GridWorld.draw(params, {labels: params.labels, trajectory : trajectory});
~~~~

### Exercise
Sample some of the noisy agent's trajectories by repeatedly clicking "run". Does the agent ever fall down the hill? Why not? By varying the softmax noise `alpha` and other parameters, find a setting where the agent's modal trajectory length is the same as the `totalTime`. (Don't modify the `transitionNoiseProb` parameter for this exercise). 
<!-- let alpha=0.5 and action cost = -.01 -->


## Hiking with stochastic transitions

When softmax noise is high, the agent will make many small "mistakes" (i.e. suboptimal actions given the agent's own preferences) but few large mistakes. In contrast, sources of noise in the environment will change the agent's state transitions independent of the agent's preferences. In the hiking example, imagine that the weather is very wet and windy. As a result, Alice will sometimes intend to go one way but actually go another way (because she slips in the mud). In this case, the shorter route to the peaks might be too risky for Alice.

To model bad weather, we assume that at every timestep, there is a constant independent probability `transitionNoiseProb` of the agent moving orthogonally to their intended direction. The independence assumption is unrealistic (if a location is slippery at one timestep it's more likely slippery the next) but is simple and satisfies the Markov assumption.

Setting `transitionNoiseProb=0.1`, the agent's first intended action is "up" instead of "right", because the shorter route is risky. 

### Exercise
Keeping `transitionNoiseProb=0.1`, find settings for the arguments to `makeHike` such that the agent goes "right" instead of "up". <!-- put up timeCost to -.5 or so --> 

~~~~
// parameters for building Hiking MDP
var utilityTable = {east:10, west:1, hill:-10, timeCost:-.1};
var startState = [0,1];

var alpha = 100;
var transitionNoiseProb = 0.1;
var totalTime = 12;
var numRejectionSamples = 1;
var params = makeHike(transitionNoiseProb, alpha, utilityTable);
var out = sample(mdpSimulateGridworld(startState, totalTime, params, numRejectionSamples));
GridWorld.draw(params, {labels: params.labels, trajectory : out});
~~~~

In a world with stochastic transitions, the agent sometimes finds itself in a state it did not intend to reach. The functions `agent` and `expUtility` (inside `mdpSimulateGridworld`) implicitly compute the expected utility of actions for every possible future state -- including states that the agent will try to avoid. In the MDP literature, this function from state-time pairs to actions (or distributions on actions) is called a *policy*. (For infinite horizon MDPs, policies are functions from states to actions, which makes them somewhat simpler to think about.) 

The example above showed that the agent chooses the long route (steering clear of the steep hill). At the same time, the agent computes what to do if it ends up moving right on the first action. (The code below doesn't prove this; it just illustrates what the agent would do it moved right.)

~~~~
// parameters for building Hiking MDP
var utilityTable = {east:10, west:1, hill:-10, timeCost:-.1};

// Change start state from [1,0] to [1,1] and reduce time
var startState = [1,1];
var totalTime = 12 - 1;
var numRejectionSamples = 1;

var alpha = 100;
var transitionNoiseProb = 0.1;
var params = makeHike(transitionNoiseProb, alpha, utilityTable);
var out = sample(mdpSimulateGridworld(startState, totalTime, params, numRejectionSamples));
GridWorld.draw(params, {labels: params.labels, trajectory : out});
~~~~

Extending this idea, we can output and visualize the expected values of actions the agent *could have taken* during their trajectory. For each state in a trajectory, we compute the expected value of each possible action (given the state and the time remaining). The resulting numbers are analogous to Q-values in infinite-horizon MDPs. 

The expected values we seek to display are already being computed: we add a function addition to `mdpSimulateGridworld` in order to output them.




~~~~

var mdpSimulateGridworld = function(startState, totalTime, params, numRejectionSamples){
  var alpha = params.alpha;
  var transition = params.transition;
  var utility = params.utility;
  var actions = params.actions;
  var isTerminal = function(state){return state[0]=='dead';};

    
  var agent = dp.cache(function(state, timeLeft){
    return Enumerate(function(){
        var action = uniformDraw(actions);
        var eu = expUtility(state, action, timeLeft);    
        factor( alpha * eu);
        return action;
        });      
    });
  
  
  var expUtility = dp.cache(function(state, action, timeLeft){
    var u = utility(state,action);
    var newTimeLeft = timeLeft - 1;
    
    if (newTimeLeft == 0 | isTerminal(state)){
      return u; 
    } else {                     
      return u + expectation( Enumerate(function(){
        var nextState = transition(state, action); 
        var nextAction = sample(agent(nextState, newTimeLeft));
        return expUtility(nextState, nextAction, newTimeLeft);  
      }));
    }                      
  });
  
  var simulate = function(startState, totalTime){
  
    var sampleSequence = function(state, timeLeft){
      if (timeLeft == 0 | isTerminal(state)){
        return [];
      } else {
      var action = sample(agent(state, timeLeft));
        var nextState = transition(state,action); 
        return [[state,action]].concat( sampleSequence(nextState,timeLeft-1 ))
      }
    };
    return Rejection(function(){return sampleSequence(startState, totalTime);}, numRejectionSamples);
  };


// Additions for outputting expected values
  var downToOne = function(n){
    return n==0 ? [] : [n].concat(downToOne(n-1));
  };

// TODO: we're currently just taking the MAP of a single sample from rejection
  var getExpUtility = function(){
    var erp = simulate(startState, totalTime);
    var states = map(first, erp.MAP().val); // go from [[state,action]] to [state]
    var timeStates = zip(downToOne(states.length), states); // [ [timeLeft, state] ] for states in trajectory

    // compute expUtility for each pair of form [timeLeft,state]
    return map( function(timeState){
      var timeLeft = timeState[0];
      var state = timeState[1];
      return [state, map(function(action){
        return expUtility(state, action, timeLeft);
      }, params.actions)];
    }, timeStates);
  };
  
  // mdpSimulateGridworld now returns both an ERP over trajectories and
  // the expUtility values for MAP trajectory
  return {erp: simulate(startState, totalTime),
          stateToExpUtilityLRUD:  getExpUtility()};

};



// long route better and takes long route
var noiseProb = .03;
var alpha = 200;
var utilityTable = {east:10, west:7, hill:-40, timeCost:-.4};

var params = makeHikeBig(noiseProb, alpha, utilityTable);

var totalTime = 12;
var startState = [1,1];
var out = mdpSimulateGridworld(startState, totalTime, params, 1);
var trajectoryExpUtilities = out.stateToExpUtilityLRUD;
GridWorld.draw(params, {labels: params.labels,
                        trajectory : trajectoryExpUtilities, 
                        expUtilities : trajectoryExpUtilities});


// TODO FIX: stochastic expUtilities and doesnt take highest EU
// action despite low noise. 
var noiseProb = .04;
var alpha = 100;
var utilityTable = {east:15, west:7, hill:-40, timeCost:-.8};

var params = makeHikeBig(noiseProb, alpha, utilityTable);

var totalTime = 14;
var startState = [1,1];
var out = mdpSimulateGridworld(startState, totalTime, params, 1);
var trajectoryExpUtilities = out.stateToExpUtilityLRUD;
print(sample(out.erp));
print('exp   ' + trajectoryExpUtilities);
GridWorld.draw(params, {labels: params.labels,
                        trajectory : trajectoryExpUtilities, 
                        expUtilities : trajectoryExpUtilities});

~~~~


--------------

[Table of Contents](/)
