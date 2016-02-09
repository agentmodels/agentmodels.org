---
layout: chapter
title: "Building agent models in WebPPL"
description: "Introduction to building agents- starting from simple but extendable agent programs." 
is_section:true
---


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



## One-shot decisions: stochastic (random) world and random actions
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

The `inferAgent`, which uses the "planning-as-inference" idiom, can also be extended using `expectation`. Previously, the action of the `inferAgent` was conditioned on its leading to the best outcome ("pizza"). This time, Tom is not aiming to choose the action most likely to have the best outcome. Instead, he wants the action with better outcomes in average. This can be represented in `inferAgent` by switching from a `condition` statement to a `factor` statement. The `condition` statement expresses a "hard" constraint on actions: actions that fail the condition are completely ruled out. The `factor` statement expresses a "soft" condition: the input to `factor` for an action is added to the action's log-score. 

To illustrate `factor`, consider this variant of the `twoHeads` example above. Instead of placing a hard constraint on the total number of heads, we increase the log-score of each possible return value (`a & b` can be `true` or `false`) by adding the number of heads. This results in a probability for `a & b` of

$$
\frac {e^{2}} { (e^{0} + 2e^{1}) + e^{2} }
$$. 

~~~~
var softHeads = Enumerate(function(){
  var a = flip(0.5);
  var b = flip(0.5);
  factor( a + b );
  return a & b;
});
viz.print(twoHeads);
~~~~

Applying the same idea to the `inferAgent`, we obtain the `softmaxAgent`:

~~~~
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

var alpha = 1;

var softmaxAgent = function(state){
  return Enumerate(function(){
      
    var action = uniformDraw(['french', 'italian']);
      
    var EU = function(action){
      return expectation( Enumerate( function(){
        return utility(transition(state,action));
      }));
    };
    factor( alpha*EU(action) )  
    return action;
  })
};

viz.print(softmaxAgent('default');
~~~~

The `softmaxAgent` differs in two ways from the `maxEUAgent` above. First, it uses the planning-as-inference idiom. Second, it does not deterministically choose the maximal expected utility action. Instead, it implements *soft* maximization, selecting actions with a probability depending on their expected utility. Formally, let the agent's probability of choosing action be $$C(a;s)$$ for $$a \in A$$ and in $$s \in S$$. Then the *softmax* decision rule is:

$$
C(a; s) \propto e^{\alpha E(U(T(s,a))) }
$$

The noise parameter $$\alpha$$ modulates between random choice $$(\alpha=0)$$ and the perfect maximization $$(\alpha = \infty)$$ of the `maxEUAgent`.

Since rational agents will *always* take the best action, why consider softmax agents? If the task is to provide normative advice on how to solve a one-shot decision problem, then "hard" maximization is the way to go. An important goal for this tutorial is to infer the preferences and beliefs of agents from their choices. These agents might not always choose the normatively optimal actions. The softmax agent provides a computationally simple, analytically tractable model of suboptimal choices. This model has been tested empirically on human action selection [cite saccades, luce choice]. Moreover, it has been used extensively in Inverse Reinforcement Learning as a model of human errors (cambridge turn taking dialogue, taxi cab paper). For for this reason, we employ the softmax model throughout this tutorial. When modeling an agent assumed to be optimal, the noise parameter $$\alpha$$ can be chosen to approximation hard maximization. 

[footnote:  One normative reason to consider softmax agents, which we won't pursue in this tutorial, is for *exploration* in the setting of reinforcement learning (link).]


