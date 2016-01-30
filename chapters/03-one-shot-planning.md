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

## One-shot decisions: deterministic actions
In a decision problem, an agent must choose between a set of actions. The agent will try to choose the action that is best in terms of their own preferences. This usually depends only on the *consequences* of the action. So the agent will try to pick actions with preferable consequences.

For example, suppose Tom is choosing between restaurants and he cares only about getting pizza. There's an Italian restaurant and a French restaurant. So John will choose the Italian restaurant because it leads to the state where gets pizza.

Formally, Tom selects an action $$a \in A$$ from the set of actions (one action for each restaurant). The consequences of an action are represented by a transition function $$T:(S,A) \to S$$ from state-action pairs to states. In our example, the relevant states are whether or not Tom gets pizza. Tom's preferences are represented by a real-valued utility function $$U:S \to R$$, which indicates the relative goodness of each state. 

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

There is an alternative way to compute the optimal action for this problem. The idea is to treat planning and decision making as an inference problem. The previous chapter showed how we can infer the probability that a coin landed Heads, given the observation that two of three coins were Heads. 

~~~~
var twoHeads = Enumerate(function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  condition( a + b + c == 2 );
  return a;
});
print(twoHeads);  
~~~~

We now use the same inference machinery to compute the optimal action in Tom's decision problem:

~~~~
var argMax = function(f,ar){return maxWith(f,ar)[0]};

var actions = ['italian', 'french'];
  
var transition = function(state, action){
    return action=='italian' ? 'pizza' : 'no pizza';
};
  
var inferAgent = function(state){
    return Enumerate(function(){
        var action = uniformDraw(actions);
        condition( transition(action) === 'pizza' );
        return action;
    });
};

inferAgent("default");

~~~~
