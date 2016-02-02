---
layout: chapter
title: "Sequential decision problems (MDPs)"
description: Mathematical framework, implementation in WebPPL with explicit recursion (could compare to value iteration), Gridworld examples.

---


PLAN:

1. We only consider very simple problems. In the following chapters, we consider more complex problems of two kinds. First, we look at sequential decision making. Cases where the selection of action now depends on how the agent will choose actions in the future. These problems are still simple in that they only involve a single agent. In later chapters, we consider game-theoretic situations involving multiple agents. 

2. Introduce MDP. Give example of moving round city to choose restaurants. To eat at a restaurant, need to first walk to it. Also prefer a shorter to longer route. Show example. 

3. Go through math of this case. Restaurant example can be solved by shortest path type algorithms. But for probabilistic case we can't do that.

4. Discounting example. Two summits. Might be unknown which is more preferred (if you just have satellite image and movement data). Cliff is just a steep hill that would hurt if you fell down (and probably end the hike). Could think of graph more abstractly: cliff as states that you reach if you take a very fast route (or if there's a route with worse heights, you might get vertigo and have to stop). 

good to think about what's stochastic in restaurant street example. attending to tempting thigns might be.

also good to think about andreas example of infinite time horizon but with small probability of death at each age (similar to language models with prob of infinite lenght sentence). 


## Sequential Decision Problems: Introduction
The previous [chapter](/chapters/03-one-shot-planning) introduced agent models for solving very simple decision problems. The rest of the tutorial looks at more complex and interesting problems. Later chapters will look at problems where the outcome depends on the decison of another rational agent (as in *game theory*). The next few chapters look at single-agent problems that are *sequential* rather than *one-shot*. In sequential decision problems, an agent's choice of action *now* depends on the action they'll choose in the future. (Agents must *co-ordinate* with their future selves).

## Markov Decision Process (MDP): example
As a simple illustration of a sequential decision problem, suppose that an agent, Alice, is looking for somewhere to eat. Alice gets out of work in a particular location (labeled "start"). She knows the streets and the restaurants nearby. Her decision problem is to take a sequence of actions such that (a) she eats at a restaurant she likes, (b) she does not spend too much time walking. Here is a visualization of the street layout, including Alice's starting location and the nearby restaurants.

[ depiction of restaurant gridworld.
Gridworld. Variant on the restaurant "donut" domain. (Because probably we don't want people distracted by those features). Could have a similar loop, but on left rather than right. Maybe make the two ways to Veg Cafe pretty close. 
]

## MDP: formal defition
We represent Alice's decision problem as a Markov Decision Process (MDP) and specifically as a discrete "Gridworld" environment. An MDP is characterized by a tuple $$(S,A(s),T(s,a),U(s,a))$$, including the *states*, the *actions* in each state, the *transition function*, and the *utility* or *reward* function. In our example, the states $$S$$ are Alice's locations on the grid. At each state, Alice selects an action $$a \in {up, down, left, right}$$, which move Alice around the grid (according to transition function $$T$$). We assume that Alice's actions, as well as the transitions and utilities of restaurants, are all deterministic. However, we will describe an agent model (and implementation in WebPPL) that solves the general case of an MDP with stochastic transitions, actions and rewards.

[Sidenote: The problem is called a "Markov Decision Process" because the environment it describes satisfies the *Markov assumption*. That is, the current state $$s \in S$$ fully characterizes the distribution on rewards and the conditional distribution on state transitions given actions.]

As with the one-shot decisions of the previous chapter, the agent in an MDP will choose actions that maximize expected utility. This depends on the total utility over the sequence of states the agent visits. Formally, let $$EU_{s}[a]$$ be the expected (total) utility of action $$a$$ in state $$s$$. The agent's choice is a softmax function of this expected utility:

$$
C(a; s) \propto e^{\alpha EU_{s}[a]}
$$

The expected utility depends (recursively) on the current and the future utility: 

$$
EU_{s}[a] = U(s, a) + E_{s', a'}(EU_{s'}[a'])
$$

with the next state $$s' \sim T(s,a)$$ and $$a' \sim C(s')$$. The decision problem ends either when a *terminal* state is reached or when the time-horizon is reached. (In the next few chapters, the time-horizon will always be finite). 

The intuition to keep in mind for MDPs is the that expected utility will propagate backwards from possible future states to the current action. If a high utility state can be reached by a sequence of actions starting from action $$a$$, then that action will have high expected utility -- *provided* that the sequence of actions is taken with high probability and there are no low utility steps along the way.


## MDPs: implementation
The recursive decision rule for MDP agents [reference?] can be directly translated into WebPPL. The resulting agent model is also a natural extension of the `softmaxAgent` from the previous [chapter](/chapters/03-one-shot-planning). The `agent` function takes the agent's state, evaluates the expectation of actions in the state, and returns a softmax distribution over actions. The expected utility of an action is computed by a separate function `expUtility`. Since an action's expected utility depends on the agent's future actions, `expUtility` calls `agent` in a mutual recursion, bottoming out when a terminal state is reached or when time runs out. 

We illustrate this agent model with a trivial MDP, where states are integers and actions are movements up or down the integers. In the next section we return to Alice's choice of restaurants. 

~~~~
var transition = function(state, action){
  return state + action;
};
  
var utility = function(state){
  return state==3 ? 1 : 0;
};

var agent = function(state, timeLeft){
  return Enumerate(function(){
    var action = uniformDraw([-1,1]);
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

The function `displayGrid` visualizes a gridworld MDP.

~~~~
var noiseProb = 0;
var alpha = 100;
var params = makeDonut(noiseProb, alpha);
var startState = [2,0];

displayGrid(params);
// TODO display this properly in gridworld (with no agent path -- just start state)
~~~~

The function `displaySequence` displays the agent's trajectory. 


~~~~
var noiseProb = 0;
var alpha = 100;
var params = makeDonut(noiseProb, alpha);
var transition = params.transition;
var utility = params.utility;
var actions = params.actions;
var isTerminal = function(state){return state[0]=='dead';};

var displaySequence = function( stateActions, params ){
  return GridWorld.zipToDisplayGrid( stateActions, params.xLim, params.yLim, true )
};

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

displaySequence( stateActionPairs, params);
// TODO display this in gridworld

print(map(function(stateAction){
  return JSON.stringify(stateAction[0]) + stateAction[1];
  }, stateActionPairs));
  
~~~~



















~~~~
var element = makeResultContainer();

var world = {
  width: 200,
  height: 200,
  fromX: 10,
  fromY: 50,
  incX: 100,
  incY: 0
};

GridWorld.draw(element, world)
~~~~
