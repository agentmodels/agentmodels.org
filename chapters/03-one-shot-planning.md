---
layout: chapter
title: One-shot planning
description: Various agent models for solving one-shot decision problems. 
---

Start with one-shot planning. Choices have some consequence. We take action that has best consequences. Implement two ways.

Outcomes probabilistic: lotteries, games of chance. Take action that is best in expectation. Write down equation.

Planning as inference for softmax. Write down softmax. Not obvious why it's a normative model. But clear that it could be good model of humans or other agents. x

## Agents for simple decision problems

The goal for the next two chapters is to build up to agent models that solve decision problems that involve long sequences of actions (e.g. MDPs). We start with "one-shot" decision problems where the agent selects a single action. These problems are trivial to solve without WebPPL. The point is to illustrate the WebPPL idioms we'll use to tackle more complex problems. 

## One-shot decisions: deterministic world
In a decision problem, an agent must choose between a set of actions. The agent will try to choose the action that is best in terms of their own preferences. This usually depends only on the *consequences* of the action. So the agent will try to pick actions with preferable consequences.

For example, suppose Tom is choosing between restaurants and he cares only about getting pizza. There's an Italian restaurant and a French restaurant. So Tom will choose the Italian restaurant because it leads to the state where gets pizza.

Formally, Tom selects an action $$a \in A$$ from the set of actions (one action for each restaurant). The consequences of an action are represented by a transition function $$T \colon S \times A \to S$$ from state-action pairs to states. In our example, the relevant states are whether or not Tom gets pizza. Tom's preferences are represented by a real-valued utility function $$U \colon S \to R$$, which indicates the relative goodness of each state. 

Tom's decision rule is to take action $$a$$ such that:

$$
\max_{a \in A} U(T(s,a))
$$

An *agent* function takes a state $$s \in S$$ as input and returns an action. For this problem, we suppose Tom starts of in the state `"default"`. The first agent we consider explicitly computes the maximum utility action:

~~~~
var argMax = function(f,ar){return maxWith(f,ar)[0]};

var actions = ['italian', 'french'];
  
var transition = function(state, action){
    return action=='italian' ? 'pizza' : 'no pizza';
};
  
var utility = function(state){
    return state == 'pizza' ? 1 : 0;
};

var maxAgent = function(state){
    return argMax( function(action){return utility(transition(state, action));},
                   actions);
};

maxAgent("default");

~~~~

There is an alternative way to compute the optimal action for this problem. The idea is to treat planning and decision making as an inference problem ("Planning as inference" cite Toussaint Botvinik). The previous chapter showed how we can infer the probability that a coin landed Heads from the observation that two of three coins were Heads. 

~~~~
var twoHeads = Enumerate(function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  condition( a + b + c == 2 );
  return a;
});
viz.print(twoHeads);
~~~~

The same inference machinery can compute the optimal action in Tom's decision problem. We sample random actions with `uniformDraw` and condition on the consequence of the action being preferred. Instead of constraining a random variable with an observation, we constrain it by the desirability of an outcome. (This resembles logical planning by backwards-chaining -- link?). 

~~~~
var actions = ['italian', 'french'];
  
var transition = function(state, action){
    return action=='italian' ? 'pizza' : 'no pizza';
};
  
var inferAgent = function(state){
    return Enumerate(function(){
        var action = uniformDraw(actions);
        condition( transition(state, action) === 'pizza' );
        return action;
    });
};

viz.print(inferAgent("default"));

~~~~



## One-shot decisions: stochastic (random) world
In the previous example, the transition function from state-action pairs to states was deterministic. So the world or *environment* was deterministic. Moreover, the agent's actions were deterministic; Tom always chose the best action ("Italian"). In contrast, many examples in this tutorial will involve a stochastic world and a "soft-max" agent. 

Imagine Tom is choosing between restaurants again. This time, Tom's preferences are about the overall quality of the meal. The meal can be "bad", "good" or "spectacular". Restaurants vary in the probability of each outcome. The formal setup is mostly the same as above. The transition function now has type signature $$ T\colon S \times A \to \Delta S $$, where $$\Delta S$$ represents a distribution over states. Tom's decision rule is now to take the action $$a \in A$$ that has the highest *average* or *expected* utility, with the expectation $$E$$ taken over the probability of different successor states $$s' \sim T(s,a)$$:

$$
\max_{a \in A} E( U(T(s,a)) )
$$

To represent this in WebPPL we extend `maxAgent` using the `expectation` function, which maps an ERP with finite support to its (real-valued) expectation:

~~~~
var argMax = function(f,ar){return maxWith(f,ar)[0]};

var actions = ['italian', 'french'];
  
var transition = function(state, action){
  var nextStates = ['bad', 'good', 'spectacular'];
  if (action=='italian'){ 
    return categorical( [0.2, 0.6, 0.2], nextStates );
  } else {
    return categorical( [0.05, 0.9, 0.05], nextStates );
  };
};
  
var utility = function(state){
  var table = {bad: -10, good: 6, spectacular:8};
  return table[state];
};


var maxEUAgent = function(state){
  var EU = function(action){
    return expectation( Enumerate( function(){
      return utility(transition(state, action));
    }));
  };
  return argMax( EU, actions);
};


maxEUAgent("default");
~~~~

The `inferAgent`, which uses the "planning-as-inference" idiom, can also be extended using `expectation`. Previously, the action of the `inferAgent` was conditioned on its leading to the best outcome ("pizza"). This time, Tom is not aiming to choose the action most likely to have the best outcome. Instead, he wants the action with better outcomes in average. This can be represented in `inferAgent` by switching from a `condition` statement to a `factor` statement. The `condition` statement expresses a "hard" constraint on actions: actions that fail the condition are completely ruled out. The `factor` statement expresses a "soft" condition: the input to `factor` for an action is added to the actions log-score. 

Here are examples illustrating `condition` vs. `factor` in the normal inference setting:

~~~~
var twoHeads = Enumerate(function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  condition( a + b + c == 2 );
  return a;
});
viz.print(twoHeads);
~~~~
~~~~
var twoHeads = Enumerate(function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  factor( a + b + c );
  return a;
});
viz.print(twoHeads);
~~~~




So in `inferAgent`, instead of completely ruling out actions which don't have the best outcome, each action is 

