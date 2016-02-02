---
layout: chapter
title: "MDPs Part II: Gridworld"
description: Mathematical framework, implementation in WebPPL with explicit recursion (could compare to value iteration), Gridworld examples.

---


PLAN

Goal is to illustrate some of the key variables that we will later do inference on. These are discounting, softmax parameter, transition noise, and the preferences. 

Discounting example. Two summits. Might be unknown which is more preferred (if you just have satellite image and movement data). Cliff is just a steep hill that would hurt if you fell down (and probably end the hike). Could think of graph more abstractly: cliff as states that you reach if you take a very fast route (or if there's a route with worse heights, you might get vertigo and have to stop). 

good to think about what's stochastic in restaurant street example. attending to tempting thigns might be.

also good to think about andreas example of infinite time horizon but with small probability of death at each age (similar to language models with prob of infinite lenght sentence). 


## MDP Example: Hiking

We introduced Markov Decision Processes (MDPs) with the example of Alice moving around a city with the aim of efficiently finding a good restaurant. Later chapters, which consider agents without full knowledge or complete rationality, will return to this example. This chapter explores some of the components of MDPs that will be important throughout this tutorial. 

We introduce a new sequential decision problem that can be represented by a "gridworld" MDP. Suppose that Bill is hiking. There are two peaks nearby, denoted "West" and "East". The peaks provide different views and Bill must choose between them. South of Bill's starting position is a steep hill. Falling down the hill would result in painful (but non-fatal) injury and end the hike early.

We represent Bill's hiking problem with a gridworld similar to Alice's restaurant choice example. The peaks are terminal states, providing differing utilities. The steep hill is also a terminal state, with negative utility. We assume a negative, constant time-cost -- so Bill prefers a shorter hike. 

~~~~
var makeHike = function(noiseProb, alpha){
  var xLim = 5;
  var yLim = 5;
  var blockedStates = [ [1,2], [1,3], [3,2] ];
  var terminals = [ [0,0], [1,0], [2,0], [3,0], [4,0], [2,2], [4,2] ]; 

  var u = function(state,action){
    if (gridEqual(state,[2,2])){return 1;}
    if (gridEqual(state,[4,2])){return 10;}
    if (isPostTerminal(state)){return 0;};
    if (stateInArray(state,terminals)) {return -10;}
    return -0.05;
  };
  return makeBlockedGridParams(xLim, yLim, blockedStates, terminals, u, noiseProb, alpha);
  };

var noiseProb = 0;
var alpha = 100;
var startState = [0,1];
var params = makeHike(noiseProb, alpha);

// TODO display
displayGrid(params);
~~~~

We start with a *deterministic* transition function. This means that Bill's only risk of falling down the steep hill is due to softmax noise in his actions. With `alpha=100`, the chance of this is tiny, and so Bill will take the short route to the peaks.


~~~~
var makeHike = function(noiseProb, alpha){
  var xLim = 5;
  var yLim = 5;
  var blockedStates = [ [1,2], [1,3], [3,2] ];
  var terminals = [ [0,0], [1,0], [2,0], [3,0], [4,0], [2,2], [4,2] ]; 

  var u = function(state,action){
    if (gridEqual(state,[2,2])){return 1;}
    if (gridEqual(state,[4,2])){return 10;}
    if (isPostTerminal(state)){return 0;};
    if (stateInArray(state,terminals)) {return -10;}
    return -0.05;
  };
  return makeBlockedGridParams(xLim, yLim, blockedStates, terminals, u, noiseProb, alpha);
  };

var noiseProb = 0;
var alpha = 100;
var startState = [0,1];
var params = makeHike(noiseProb, alpha);

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
  return sampleSequence(startState, totalTime);
};

var totalTime = 12;
var out = simulate(startState, totalTime);
print(out);
//displaySequence(out);

~~~~

If we set the softmax noise parameter `alpha=0.5`, the agent will often take sub-optimal decisions. While not realistic in Bill's situation, this might better describe an intoxicated agent. Since the agent is noisy, we sample many trajectories in order to approximate the full distribution. To construct an ERP based on these samples, we use the built-in function `Rejection` which performs inference by rejection sampling. (Our goal here is not inference over trajectories and so the `Rejection` function does not need any `factor` statement in its body). In this case, when the agent takes a suboptimal action, it will take *longer* than five steps to reach the East summit. So we plot the distribution on the length of trajectories to summarize the agent's behavior. 


~~~~
var makeHike = function(noiseProb, alpha){
  var xLim = 5;
  var yLim = 5;
  var blockedStates = [ [1,2], [1,3], [3,2] ];
  var terminals = [ [0,0], [1,0], [2,0], [3,0], [4,0], [2,2], [4,2] ]; 

  var u = function(state,action){
    if (gridEqual(state,[2,2])){return 1;}
    if (gridEqual(state,[4,2])){return 10;}
    if (isPostTerminal(state)){return 0;};
    if (stateInArray(state,terminals)) {return -10;}
    return -0.05;
  };
  return makeBlockedGridParams(xLim, yLim, blockedStates, terminals, u, noiseProb, alpha);
  };

var noiseProb = 0;
var alpha = 100;
var startState = [0,1];
var params = makeHike(noiseProb, alpha);

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
  return sampleSequence(startState, totalTime);
};

var totalTime = 12;
var numRejectionSamples = 500;
var erp = Rejection(function(){return simulate(startState,totalTime).length;},
    numRejectionSamples);
viz.print(erp);

~~~~

if we sample from the softmax agent, we can see that the agent doesn't fall down the hill, but mostly just goes into a wall or goes in the wrong direction. exercise: add enough noise that the agent often takes much longer trajectories.


add noise. now big risk of falling off. so this changes the policy, even for non-noisy agent.

decrease time: more time pressure moves you to shortcut. (as does increasing the action cost). decrease time enough and you just go to close hill (sample for increasing action cost). 


