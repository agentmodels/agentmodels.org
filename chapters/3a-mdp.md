---
layout: chapter
title: "Sequential decision problems: MDPs"
description: We look at sequential decision problems, how to formalize them mathematically, and how to implement them concisely and efficiently using probabilistic programs with mutual recursion and caching. 
---

## Introduction

The previous [chapter](/chapters/3-agents-as-programs.html) introduced agent models for solving very simple decision problems. From here on, we will tackle more interesting decision problems. Later sections will look at problems where the outcome depends on the decision of another rational agent (as in *game theory*). The next few sections will look at single-agent problems that are *sequential* rather than *one-shot*. In sequential decision problems, an agent's choice of action *now* depends on the action they will choose in the future. As in game-theoretic scenarios, the decision maker must coordinate with another rational agent. But, in sequential decision problems, that rational agent is their future self.

As a simple illustration of a sequential decision problem, suppose that an agent, Bob, is looking for a place to eat. Bob gets out of work in a particular location (indicated below by the blue circle). He knows the streets and the restaurants nearby. His decision problem is to take a sequence of actions such that (a) he eats at a restaurant he likes and (b) he does not spend too much time walking. Here is a visualization of the street layout. The labels refer to different types of restaurants: a chain selling Donuts, a Vegetarian Salad Bar and a Noodle Shop. 

~~~~
// We use the Webppl-gridworld library
var world = makeRestaurantChoiceMDP();
var startState = restaurantChoiceStart;

GridWorld.draw(world, { trajectory : [ startState ] });
~~~~


## Markov Decision Processes: Definition

We represent Bob's decision problem as a Markov Decision Process (MDP) and, more specifically, as a discrete "Gridworld" environment. An MDP is a tuple $$ \left\langle S,A(s),T(s,a),U(s,a) \right\rangle$$, including the *states*, the *actions* in each state, the *transition function* that maps state-action pairs to successor states, and the *utility* or *reward* function. In our example, the states $$S$$ are Bob's locations on the grid. At each state, Bob selects an action $$a \in \{ \text{up}, \text{down}, \text{left}, \text{right} \} $$, which moves Bob around the grid (according to transition function $$T$$). We assume that Bob's actions, as well as the transitions and utilities of the restaurants, are all deterministic. However, we will describe an agent model (and implementation in WebPPL) that solves the general case of an MDP with stochastic transitions, noisy actions, and stochastic rewards.

As with the one-shot decisions of the previous chapter, the agent in an MDP will choose actions that *maximize expected utility*. This now depends on the total utility of the *sequence* of states that the agent visits. Formally, let $$EU_{s}[a]$$ be the expected (total) utility of action $$a$$ in state $$s$$. The agent's choice is a softmax function of this expected utility:

$$
C(a; s) \propto e^{\alpha EU_{s}[a]}
$$

The expected utility depends on both immediate utility and, recursively, on future expected utility:

<br>**Expected Utility Recursion**:

$$
EU_{s}[a] = U(s, a) + \mathbb{E}_{s', a'}(EU_{s'}[a'])
$$

<br>
with the next state $$s' \sim T(s,a)$$ and $$a' \sim C(s')$$. The decision problem ends either when a *terminal* state is reached or when the time-horizon is reached. (In the next few chapters, the time-horizon will always be finite). 

The intuition to keep in mind for MDPs is that the expected utility will propagate backwards from possible future states to the current action. If a high utility state can be reached by a sequence of actions starting from action $$a$$, then action $$a$$ will have high expected utility -- *provided* that the sequence of actions is taken with high probability and there are no low utility steps along the way.


## Markov Decision Processes: Implementation

The recursive decision rule for MDP agents (Expected Utility Recursion above) can be directly translated into WebPPL. The resulting agent model is also a natural extension of the `softmaxAgent` from the previous [chapter](/chapters/3-agents-as-programs.html). The `act` function of the agent takes the agent's state, evaluates the expectation of actions in the state, and returns a softmax distribution over actions. The expected utility of an action is computed by a separate function `expectedUtility`. Since an action's expected utility depends on the agent's future actions, `expectedUtility` calls `act` in a mutual recursion, bottoming out when a terminal state is reached or when time runs out. 

We illustrate this agent model with a trival example of an MDP and return to Bob's choice of restaurant later on. The trivial MDP is implemented in WebPPL by functions `transition` and `utility`, and by the agent's available actions `[-1, 0, 1]`. The MDP is as follows:

### Integer Line MDP
- **States**: Points on the integer line (e.g -1, 0, 1, 2).

- **Actions/transitions**: Actions "left", "right" and "stay" move the agent deterministically along the line in either direction.

- **Utility**: The utility is $$1$$ for the state corresponding to the integer $$3$$ and is $$0$$ otherwise. 


Here is a WebPPL agent that starts at the origin (`state === 0`) and that takes a first step (to the right):

~~~~
var transition = function(state, action){
  return state + action;
};

var utility = function(state){
  return (state === 3) ? 1 : 0;
};

var makeAgent = function() { 
  var act = function(state, timeLeft){
    return Enumerate(function(){
      var action = uniformDraw([-1, 0, 1]);
      var eu = expectedUtility(state, action, timeLeft);    
      factor(100 * eu);
      return action;
    });      
  };

  var expectedUtility = function(state, action, timeLeft){
    var u = utility(state,action);
    var newTimeLeft = timeLeft - 1;

    if (newTimeLeft == 0){
      return u; 
    } else {                     
      return u + expectation(Enumerate(function(){
        var nextState = transition(state, action); 
        var nextAction = sample(act(nextState, newTimeLeft));
        return expectedUtility(nextState, nextAction, newTimeLeft);  
      }));
    }
  };

  return { 
    act : act
  };
}

var act = makeAgent().act;

var startState = 0;
var totalTime = 4;

// Agent's move '-1' means 'left', '0' means 'stay', '1' means 'right'
print("Agent's action: " + sample(act(startState, totalTime)));
~~~~

This code computes the agent's initial action, given that the agent will get to take four actions in total. To simulate the agent's entire trajectory, we add a third function `simulate`, which updates and stores the world state in response to the agent's actions: 

~~~~
var transition = function(state, action){
  return state + action;
};

var utility = function(state){
  return (state === 3) ? 1 : 0;
};

var makeAgent = function() { 
  var act = function(state, timeLeft){
    return Enumerate(function(){
      var action = uniformDraw([-1, 0, 1]);
      var eu = expectedUtility(state, action, timeLeft);    
      factor(100 * eu);
      return action;
    });      
  };

  var expectedUtility = function(state, action, timeLeft){
    var u = utility(state,action);
    var newTimeLeft = timeLeft - 1;

    if (newTimeLeft == 0){
      return u; 
    } else {                     
      return u + expectation(Enumerate(function(){
        var nextState = transition(state, action); 
        var nextAction = sample(act(nextState, newTimeLeft));
        return expectedUtility(nextState, nextAction, newTimeLeft);  
      }));
    }
  };

  return { act : act};
}


var act = makeAgent().act;

var simulate = function(startState, totalTime){
  var sampleSequence = function(state, timeLeft){
    if (timeLeft === 0){
      return [];
    } else {
      var action = sample(act(state, timeLeft));
      var nextState = transition(state,action); 
      return [state].concat(sampleSequence(nextState, timeLeft - 1))
    }
  };
  return sampleSequence(startState, totalTime);
};

var startState = 0;
var totalTime = 4;
print("Agent's action sequence: " + simulate(startState, totalTime));
~~~~

The `expectedUtility` and `simulate` functions are similar. The `expectedUtilty` function includes the agent's own (subjective) simulation of the future distribution on states. In the case of an MDP and an optimal agent, the agent's simulation is identical to the world simulator (up to irreducible random noise in the transition and choice functions). In later chapters, we describe agents whose subjective simulations diverge from the world simulator -- i.e. agents with inaccurate models of their future selves. 

What does the mutual recursion between `act` and `expectedUtility` look like if we unroll it? In this example, where the transition function is deterministic, there is a tree that expands until `timeLeft` reaches zero. The root is the starting state (`startState === 0`) and this branches into three successor states (`-1`, `0`, `1`). This leads to an exponential blow-up in the runtime of a single action (as a function of the number of steps taken into account when considering the future):

~~~~
// transition, utility and makeAgent functions defined exactly as above
///fold:
var transition = function(state, action){
  return state + action;
};

var utility = function(state){
  return (state === 3) ? 1 : 0;
};

var makeAgent = function() { 
  var act = function(state, timeLeft){
    return Enumerate(function(){
      var action = uniformDraw([-1, 0, 1]);
      var eu = expectedUtility(state, action, timeLeft);    
      factor(100 * eu);
      return action;
    });      
  };

  var expectedUtility = function(state, action, timeLeft){
    var u = utility(state,action);
    var newTimeLeft = timeLeft - 1;

    if (newTimeLeft == 0){
      return u; 
    } else {                     
      return u + expectation(Enumerate(function(){
        var nextState = transition(state, action); 
        var nextAction = sample(act(nextState, newTimeLeft));
        return expectedUtility(nextState, nextAction, newTimeLeft);  
      }));
    }
  };

  return { act : act};
}


var act = makeAgent().act;

var simulate = function(startState, totalTime){
  var sampleSequence = function(state, timeLeft){
    if (timeLeft === 0){
      return [];
    } else {
      var action = sample(act(state, timeLeft));
      var nextState = transition(state,action); 
      return [state].concat(sampleSequence(nextState, timeLeft - 1))
    }
  };
  return sampleSequence(startState, totalTime);
};
///

var startState = 0;

var getRuntime = function(totalTime){
  return timeit(function(){
    return act(startState,totalTime);
  }).runtimeInMilliseconds.toPrecision(4);
};

var numSteps = [3, 4, 5, 6, 7];
var runtimes = map(getRuntime, numSteps);

print('Runtime in ms for for a given number of steps: \n')
print(_.object(numSteps, runtimes));
viz.bar(numSteps, runtimes)
~~~~

Most of this computation is unnecessary. If the agent starts at `state === 0`, there are three ways the agent could be at `state === 0` again after two steps: either the agent stays put twice or the agent goes one step away and then returns. The code above computes `agent(0, totalTime-2)` three times, while it only needs to be computed once. This problem can be resolved by *memoization*, which stores the results of a function call for re-use when the function is called again on the same input. This use of memoization results in a runtime that is polynomial in the number of states and the total time. We explore the efficiency of these algorithms in more detail in Section VI. In WebPPL, we use the higher-order function `dp.cache` to memoize the `act` and `expectedUtility` functions:

~~~~
// transition, utility and makeAgent functions defined exactly as above
///fold:
var transition = function(state, action){
  return state + action;
};

var utility = function(state){
  return (state === 3) ? 1 : 0;
};

var makeAgent = function() { 
  var act = dp.cache(function(state, timeLeft){
    return Enumerate(function(){
      var action = uniformDraw([-1, 0, 1]);
      var eu = expectedUtility(state, action, timeLeft);    
      factor(100 * eu);
      return action;
    });      
  });

  var expectedUtility = dp.cache(function(state, action, timeLeft){
    var u = utility(state,action);
    var newTimeLeft = timeLeft - 1;

    if (newTimeLeft == 0){
      return u; 
    } else {                     
      return u + expectation(Enumerate(function(){
        var nextState = transition(state, action); 
        var nextAction = sample(act(nextState, newTimeLeft));
        return expectedUtility(nextState, nextAction, newTimeLeft);  
      }));
    }
  });

  return { act : act};
}


var act = makeAgent().act;

var simulate = function(startState, totalTime){
  var sampleSequence = function(state, timeLeft){
    if (timeLeft === 0){
      return [];
    } else {
      var action = sample(act(state, timeLeft));
      var nextState = transition(state,action); 
      return [state].concat(sampleSequence(nextState, timeLeft - 1))
    }
  };
  return sampleSequence(startState, totalTime);
};
///

var startState = 0;

var getRuntime = function(totalTime){
  return timeit(function(){
    return act(startState,totalTime);
  }).runtimeInMilliseconds.toPrecision(4);
};

var numSteps = [3, 4, 5, 6, 7];
var runtimes = map(getRuntime, numSteps);

print('WITH MEMOIZATION \n');
print('Runtime in ms for for a given number of steps: \n')
print(_.object(numSteps, runtimes));
viz.bar(numSteps, runtimes)
~~~~


## Choosing restaurants in Gridworld

The agent model above that includes memoization allows us to solve Bob's "Restaurant Choice" problem efficiently. 

We extend the agent model above by adding `isTerminal` to halt simulations when the agent reaches a terminal state. For the Restaurant Choice problem, the restaurants are assumed to be terminal states. After computing the agent's trajectory, we use the WebPPL [Gridworld library](https://github.com/agentmodels/webppl-gridworld) to animate it. 

~~~~
// We use the WebPPL-gridworld library

// Construct world
var world = makeRestaurantChoiceMDP();
var transition = world.transition;
var stateToActions = world.stateToActions;
var gridLocationToRestaurant = world.feature;

// Construct agent utility function
var utilityTable = {'Donut S': 1, 
                    'Donut N': 1, 
                    'Veg': 3,
                    'Noodle': 2, 
                    'timeCost': -0.1};

var tableToUtilityFunction = function(table, feature){  
  return function(state, action){
    var stateFeatureName = feature(state).name;
    return stateFeatureName ? table[stateFeatureName] : table.timeCost;
  };
};


var act = dp.cache(function(state){
  return Enumerate(function(){
    var action = uniformDraw(stateToActions(state));
    var eu = expectedUtility(state, action);
    factor(100 * eu);
    return action;
  });
});

var expectedUtility = dp.cache(function(state, action){
  var u = utility(state, action);
  
  if (state.terminateAfterAction){
    return u; 
  } else {
    return u + expectation(Enumerate(function(){
      var nextState = transition(state, action);
      var nextAction = sample(act(nextState));
      return expectedUtility(nextState, nextAction);
    }));
  }
});
    
var utility = tableToUtilityFunction(utilityTable, gridLocationToRestaurant);


// Define agent
var makeAgent = function(){
  var act = dp.cache(function(state){
    return Enumerate(function(){
      var action = uniformDraw(stateToActions(state));
      var eu = expectedUtility(state, action);
      factor(100 * eu);
      return action;
    });
  });

  var expectedUtility = dp.cache(function(state, action){
    var u = utility(state, action);

    if (state.terminateAfterAction){
      return u; 
    } else {                     
      return u + expectation(Enumerate(function(){
        var nextState = transition(state, action);
        var nextAction = sample(act(nextState));
        return expectedUtility(nextState, nextAction);  
      }));
    }
  });
  return {act:act}
};

var act = makeAgent().act;

var simulate = function(startState){
  var sampleSequence = function(state){
    var action = sample(act(state));
    var nextState = transition(state, action);
	var out = [state, action];
    return state.terminateAfterAction ? [out]
	  : [out].concat(sampleSequence(nextState));
  };
  return sampleSequence(startState);
};

var startState = {loc: [3,1],
                  timeLeft: 9,
                  timeAtRestaurant: 1};
var trajectory = simulate(startState);
GridWorld.draw(world, {trajectory : map(first, trajectory)});

~~~~

### Noisy agents, stochastic environments

This section looked at two MDPs that were essentially deterministic. Part of the difficulty of solving MDPs is that actions, rewards and transitions can be stochastic. The next [chapter](/chapters/3b-mdp-gridworld.html) explores both noisy agents and stochastic gridworld environments.

--------------

[Table of Contents](/)
