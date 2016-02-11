---
layout: chapter
title: "Sequential decision problems, part I: Setup"
description: Motivating example of sequential decision problem, MDP formalism and implementation in WebPPL with mutual recursion, exponential trees action evaluation and a resolution via memoization. 
---


## Introduction

The previous [chapter](/chapters/03-one-shot-planning) introduced agent models for solving very simple decision problems. From here on, we tackled more interesting decision problems. Later sections will look at problems where the outcome depends on the decision of another rational agent (as in *game theory*). The present sections looks at single-agent problems that are *sequential* rather than *one-shot*. In sequential decision problems, an agent's choice of action *now* depends on the action they'll choose in the future. (As in game theory, the decision maker must co-ordinate with another rational agent. But in sequential decision problems, that rational agent is one's future self).

## Markov Decision Process (MDP): example
As a simple illustration of a sequential decision problem, suppose that an agent, Bob, is looking for somewhere to eat. Bob gets out of work in a particular location (indicated below by the blue circle). He knows the streets and the restaurants nearby. His decision problem is to take a sequence of actions such that (a) he eats at a restaurant he likes, (b) he does not spend too much time walking. Here is a visualization of the street layout. The labels refer to different types of restaurant: a chain selling Donuts, a Vegetarian Salad Bar and a Noodle Shop. 

~~~~
// We use functions from the WebPPL-gridworld library, which we'll explain later
var params = makeDonutInfer(true, {'donutSouth': 1, 'donutNorth': 1, 'veg': 1,
                                   'noodle': 1, 'timeCost': -0.1}, 100, 0);
GridWorld.draw(params, {labels: params.labels, trajectory: [[[2,0]]]});
~~~~


## MDP: formal definition
We represent Bob's decision problem as a Markov Decision Process (MDP) and specifically as a discrete "Gridworld" environment. An MDP is characterized by a tuple $$(S,A(s),T(s,a),U(s,a))$$, including the *states*, the *actions* in each state, the *transition function*, and the *utility* or *reward* function. In our example, the states $$S$$ are Bob's locations on the grid. At each state, Bob selects an action $$a \in \{ \text{up}, \text{down}, \text{left}, \text{right} \} $$, which moves Bob around the grid (according to transition function $$T$$). We assume that Bob's actions, as well as the transitions and utilities of restaurants, are all deterministic. However, we will describe an agent model (and implementation in WebPPL) that solves the general case of an MDP with stochastic transitions, noisy actions and stochastic rewards.

As with the one-shot decisions of the previous chapter, the agent in an MDP will choose actions that *maximize expected utility*. This now depends on the total utility over the *sequence* of states the agent visits. Formally, let $$EU_{s}[a]$$ be the expected (total) utility of action $$a$$ in state $$s$$. The agent's choice is a softmax function of this expected utility:

$$
C(a; s) \propto e^{\alpha EU_{s}[a]}
$$

The expected utility depends (recursively) on the current and the future utility:

$$
EU_{s}[a] = U(s, a) + E_{s', a'}(EU_{s'}[a'])
$$

with the next state $$s' \sim T(s,a)$$ and $$a' \sim C(s')$$. This equation for expected utility will be denoted **Equation 1**. The decision problem ends either when a *terminal* state is reached or when the time-horizon is reached. (In the next few chapters, the time-horizon will always be finite). 

The intuition to keep in mind for MDPs is that the expected utility will propagate backwards from possible future states to the current action. If a high utility state can be reached by a sequence of actions starting from action $$a$$, then action $$a$$ will have high expected utility -- *provided* that the sequence of actions is taken with high probability and there are no low utility steps along the way.


## MDPs: implementation
The recursive decision rule for MDP agents (Equation 1) can be directly translated into WebPPL. The resulting agent model is also a natural extension of the `softmaxAgent` from the previous [chapter](/chapters/03-one-shot-planning). The `agent` function takes the agent's state, evaluates the expectation of actions in the state, and returns a softmax distribution over actions. The expected utility of an action is computed by a separate function `expUtility`. Since an action's expected utility depends on the agent's future actions, `expUtility` calls `agent` in a mutual recursion, bottoming out when a terminal state is reached or when time runs out. 

We illustrate this agent model with a trival example of an MDP and return to Bob's choice of restaurant later on. The trivial MDP is implemented in WebPPL by functions `transition` and `utility`, and by agents available actions `[-1, 1]`. The MDP is as follows:

### Integer Line MDP
- **States**: Points on the integer line (e.g -1, 0, 1, 2).

- **Actions/transitions**: Actions "left" and "right" move agent deterministically long the line in either direction.

- **Utility**: We have `utility(3)==1` and zero elsewhere. 


Here is a WebPPL agent that solves this problem:

~~~~
var transition = function(state, action){
  return state + action;
};
  
var utility = function(state){
  return state==3 ? 1 : 0;
};

var agent = function(state, timeLeft){
  return Enumerate(function(){
    var action = uniformDraw([-1, 1]);
    var eu = expUtility(state, action, timeLeft);    
    factor(100 * eu);
    return action;
  });      
};

var expUtility = function(state, action, timeLeft){
  var u = utility(state,action);
  var newTimeLeft = timeLeft - 1;
  
  if (newTimeLeft == 0){
    return u; 
  } else {                     
    return u + expectation( Enumerate(function(){
      var nextState = transition(state, action); 
      var nextAction = sample(agent(nextState, newTimeLeft));
      return expUtility(nextState, nextAction, newTimeLeft);  
    }));
  }                      
};

var startState = 0;
var totalTime = 4;
viz.print(agent(startState, totalTime));

// TODO: could try a gridworld here for line environment. could just make a gridworld
// of form [.... -1,0,1,2,3,4 ...]. no terminal states and only state 3 has utility 1 (rest
// zero). 

~~~~

This code computes the agent's initial action, where the agent starts at `state=0` and assumes four actions will be taken total. To simulate the agent's entire trajectory, we add a third function `simulate`, which updates and stores the world state given the agent's action.


~~~~
var transition = function(state, action){
  return state + action;
};
  
var utility = function(state){
  return state==3 ? 1 : 0;
};

var agent = function(state, timeLeft){
  return Enumerate(function(){
    var action = uniformDraw([-1,0,1]);
    var eu = expUtility(state, action, timeLeft);    
    factor(100 * eu);
    return action;
  });
};

var expUtility = function(state, action, timeLeft){
  var u = utility(state,action);
  var newTimeLeft = timeLeft - 1;
  
  if (newTimeLeft == 0){
    return u; 
  } else {
    return u + expectation( Enumerate(function(){
      var nextState = transition(state, action); 
      var nextAction = sample(agent(nextState, newTimeLeft));
      return expUtility(nextState, nextAction, newTimeLeft);  
    }));
  }
};

var simulate = function(startState, totalTime){
  
  var sampleSequence = function(state, timeLeft){
    if (timeLeft == 0){
      return [];
    } else {
      var action = sample(agent(state, timeLeft));
      var nextState = transition(state,action); 
      return [state].concat( sampleSequence(nextState,timeLeft-1 ))
    }
  };
  return sampleSequence(startState, totalTime);
};

var startState = 0;
var totalTime = 4;
print(simulate(startState, totalTime));

~~~~

The `expUtility` and `simulate` functions are similar. The `expUtilty` function includes the agent's own (subjective) simulation of the future distribution on states. In the case of an MDP and an optimal agent, the agent's simulation is identical to the world simulator (up to irreducible random noise in the the transition and choice functions). In later chapters, the agent's subjective simulations will diverge from the world simulator and become inaccurate.

What does the mutual recursion between `agent` and `expUtility` look like if we unpack it? In this example, where the transition function is deterministic, there is a tree that expands up until `timeLeft` reaches zero. The root is the starting state and this branches into three successor states. This leads to an exponential blow-up in the runtime of a single action:

~~~~
var transition = function(state, action){
  return state + action;
};
  
var utility = function(state){
  return state==3 ? 1 : 0;
};

var agent = function(state, timeLeft){
  return Enumerate(function(){
    var action = uniformDraw([-1,0,1]);
    var eu = expUtility(state, action, timeLeft);    
    factor(100 * eu);
    return action;
  });
};

var expUtility = function(state, action, timeLeft){
  var u = utility(state,action);
  var newTimeLeft = timeLeft - 1;
  
  if (newTimeLeft == 0){
    return u; 
  } else {                     
    return u + expectation( Enumerate(function(){
      var nextState = transition(state, action); 
      var nextAction = sample(agent(nextState, newTimeLeft));
      return expUtility(nextState, nextAction, newTimeLeft);
    }));
  }
};

var startState = 0;

var getRuntime = function(totalTime){
    return timeit( function(){
        return agent(startState,totalTime);
    }).runtimeInMilliseconds;
};

var totalTimes = [3, 4, 5, 6, 7];
print('Runtime in ms for total times: ' + totalTimes + '\n' +
    map(getRuntime, totalTimes) );

~~~~


Most of this computation is unnecessary. If the agent starts at `state=0`, there are three ways the agent could be at `state=0` again at `timeLeft=totalTime-2`: either the agent stays put twice or the agent away and then returns. The current code will compute `agent(0,totalTime-2)` three times, while it only needs to be computed once. This problem can be resolved by *memoization* (via the `dp.cache` function), which stores the results of a function call so they can be re-used if the function is called again on the same input. In general, memoization means the runtime is polynomial in the number of states and the total time. 

~~~~
var transition = function(state, action){
  return state + action;
};
  
var utility = function(state){
  return state==3 ? 1 : 0;
};

var agent = dp.cache(function(state, timeLeft){
  return Enumerate(function(){
    var action = uniformDraw([-1,0,1]);
    var eu = expUtility(state, action, timeLeft);    
    factor(100 * eu);
    return action;
  });
});

var expUtility = dp.cache(function(state, action, timeLeft){
  var u = utility(state,action);
  var newTimeLeft = timeLeft - 1;
  
  if (newTimeLeft == 0){
    return u; 
  } else {                     
    return u + expectation( Enumerate(function(){
      var nextState = transition(state, action); 
      var nextAction = sample(agent(nextState, newTimeLeft));
      return expUtility(nextState, nextAction, newTimeLeft);
    }));
  }
});

var startState = 0;

var getRuntime = function(totalTime){
    return timeit( function(){
        return agent(startState,totalTime);
    }).runtimeInMilliseconds;
};

var totalTimes = [3, 4, 5, 6, 7];
print('Runtime in ms for total times: ' + totalTimes + '\n' +
    map(getRuntime, totalTimes) );
~~~~



## MDPs: Gridworld and choosing restaurants

With this agent model including memoization, we can solve Alice's restaurant choice problem efficiently. To construct the MDP we use the library "webppl-gridworld" (link), which we'll use throughout this book. The next chapter discusses this library in more detail.

We extend the code above by adding checks for whether a state is terminal. If Alice reaches a restaurant, she receives the restaurant's utility score and the decision problem ends.

The function `GridWorld.draw` visualizes a gridworld MDP.

~~~~
var noiseProb = 0;
var alpha = 100;
var params = makeDonut(noiseProb, alpha);
var startState = [2,0];

GridWorld.draw(params);
~~~~

Adding the optional parameter `{trajectory : stateActionPairs}` displays the agent's trajectory. 


~~~~
var noiseProb = 0;
var alpha = 100;
var params = makeDonut(noiseProb, alpha);
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

var startState = [2,0];
var totalTime = 7;
var stateActionPairs = simulate(startState, totalTime);

GridWorld.draw(params, {trajectory : stateActionPairs});
~~~~

--------------

[Table of Contents](/)
