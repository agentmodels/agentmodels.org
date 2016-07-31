---
layout: chapter
title: "Agents as probabilistic programs"
description: "WebPPL Agents for solving simple, one-shot decision problems" 
is_section: true
---

## Introduction

The goal for this section is to implement agents that compute rational *policies*. We can think of policies as *plans* for achieving good outcomes in situations where:

- The agent makes a *sequence* of *distinct* choices, rather than choosing once or playing the same game repeatedly (as in multiple rounds of roulette). 

- The environment is *stochastic* and so the agent's plans must take into account unlikely contingencies (e.g. avoiding a series of actions that has unlikely but calamitous risks). 

- The environment contains features which are not stochastic, but which are initially *unknown* to the agent. The agent's plans should value gaining information by *observation* that will facilitate better future plans. 

As a concrete example, consider navigating a foreign city to efficiently find a good coffee shop.

To build up to agents that form such policies, we start with agents that solve the very simplest decision problems. These are *one-shot* problems, where the agent selects a single action (not a sequence of actions). The problems can be trivially solved with pen and paper. We use WebPPL to solve them in order to illustrate the core concepts and idioms that we will use to tackle more complex problems. 


## One-shot decisions in a deterministic world

In a *one-shot decision problem* an agent makes a single choice between a set of *actions*, each of which has potentially distinct *consequences*. A rational agent chooses the action that is best in terms of his or her own preferences. Often, this depends not on the *action* itself being preferred, but only on its *consequences*. 

For example, suppose Tom is choosing between restaurants and all he cares about is eating pizza. There's an Italian restaurant and a French restaurant. Tom would be quite happy to choose the French restaurant if it offered pizza. Since it does *not* offer pizza, Tom will choose the Italian (which does).

While this problem is trivial, we formalize it to illustrate the notation. We suppose Tom selects an action $$a \in A$$ from the set of all actions. The actions in this case are {"eat at Italian restaurant", "eat at French restaurant"}. The consequences of an action are represented by a transition function $$T \colon S \times A \to S$$ from state-action pairs to states. In our example, the relevant *state* is whether or not Tom gets his pizza. Tom's preferences are represented by a real-valued utility function $$U \colon S \to \mathbb{R}$$, which indicates the relative goodness of each state. 

Tom's *decision rule* is to take action $$a$$ that maximizes utility, i.e., the action

$$
{\arg \max}_{a \in A} U(T(s,a))
$$

In WebPPL, we can implement this utility-maximizing agent as a function `maxAgent` that takes a state $$s \in S$$ as input and returns an action. For Tom's choice between restaurants, we assume that the agent starts off in a state `"default"`, denoting whatever Tom does before going off to eat. The program directly translates the decision rule above using the higher-order function `argMax`.

~~~~
// Choose to eat at the Italian or French restaurants
var actions = ['italian', 'french'];

var transition = function(state, action){
  return (action === 'italian') ? 'pizza' : 'steak frites';
};

var utility = function(state){
  return (state === 'pizza') ? 1 : 0;
};

var maxAgent = function(state){
  return argMax(
    function(action){
      return utility(transition(state, action));
    },
    actions);
};

print("Choice in default state: " + maxAgent("default"));
~~~~

There is an alternative way to compute the optimal action for this problem. The idea is to treat choosing an action as an *inference* problem. The previous chapter showed how we can *infer* the probability that a coin landed Heads from the observation that two of three coins were Heads. 

~~~~
var twoHeads = Infer({ method: 'enumerate' }, function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  condition(a + b + c === 2);
  return a;
});

viz.auto(twoHeads);
~~~~

The same inference machinery can compute the optimal action in Tom's decision problem. We sample random actions with `uniformDraw` and condition on the preferred outcome happening. Intuitively, we imagine observing the consequence we prefer (e.g. pizza) and then *infer* from this the action that caused this consequence. <!-- address evidential vs causal decision theory? -->

This idea is known as "planning as inference" refp:botvinick2012planning. It also resembles the idea of "backwards chaining" in logical inference and planning. The `inferenceAgent` solves the same problem as `maxAgent`, but uses planning-as-inference: 

~~~~
var actions = ['italian', 'french'];

var transition = function(state, action){
  return (action === 'italian') ? 'pizza' : 'steak frites';
};

var inferenceAgent = function(state){
  return Infer({ method: 'enumerate' }, function(){
    var action = uniformDraw(actions);
    condition(transition(state, action) === 'pizza');
    return action;
  });
};

viz.auto(inferenceAgent("default"));
~~~~


## One-shot decisions in a stochastic world

In the previous example, the transition function from state-action pairs to states was *deterministic* and so described a deterministic world or environment. Moreover, the agent's actions were deterministic; Tom always chose the best action ("Italian"). In contrast, many examples in this tutorial will involve a *stochastic* world and a noisy "soft-max" agent. 

Imagine that Tom is choosing between restaurants again. This time, Tom's preferences are about the overall quality of the meal. A meal can be "bad", "good" or "spectacular" and restaurants vary in the probability with which they produce each level of quality. The formal setup is mostly as above. The transition function now has type signature $$ T\colon S \times A \to \Delta S $$, where $$\Delta S$$ represents a distribution over states. Tom's decision rule is now to take the action $$a \in A$$ that has the highest *average* or *expected* utility, with the expectation $$\mathbb{E}$$ taken over the probability of different successor states $$s' \sim T(s,a)$$:

$$
\max_{a \in A} \mathbb{E}( U(T(s,a)) )
$$

To represent this in WebPPL, we extend `maxAgent` using the `expectation` function, which maps a distribution with finite support to its (real-valued) expectation:

~~~~
var actions = ['italian', 'french'];

var transition = function(state, action){
  var nextStates = ['bad', 'good', 'spectacular'];
  if (action === 'italian'){ 
    return categorical([0.2, 0.6, 0.2], nextStates);
  } else {
    return categorical([0.05, 0.9, 0.05], nextStates);
  };
};

var utility = function(state){
  var table = { bad: -10, good: 6, spectacular: 8 };
  return table[state];
};

var maxEUAgent = function(state){
  var expectedUtility = function(action){
    return expectation(Infer({ method: 'enumerate' }, function(){
      return utility(transition(state, action));
    }));
  };
  return argMax(expectedUtility, actions);
};

maxEUAgent("default");
~~~~

The `inferenceAgent`, which uses the planning-as-inference idiom, can also be extended using `expectation`. Previously, the agent's action was conditioned on leading to the best consequence ("pizza"). This time, Tom is not aiming to choose the action most likely to have the best outcome. Instead, he wants the action with better outcomes on average. This can be represented in `inferenceAgent` by switching from a `condition` statement to a `factor` statement. The `condition` statement expresses a "hard" constraint on actions: actions that fail the condition are completely ruled out. The `factor` statement, by contrast, expresses a "soft" condition. Technically, `factor(x)` adds `x` to the unnormalized log-probability of the program execution within which it occurs.

To illustrate `factor`, consider the following variant of the `twoHeads` example above. Instead of placing a hard constraint on the total number of Heads outcomes, we give each setting of `a`, `b` and `c` a *score* based on the total number of heads. The score is highest when all three coins are Heads, but even the "all tails" outcomes is not ruled out completely.

~~~~
var softHeads = Infer({ method: 'enumerate' }, function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  factor(a + b + c);
  return a;
});

viz.auto(softHeads);
~~~~

As another example, consider the following short program:

~~~~
var y = Infer({ method: 'enumerate' }, function(){
  var n = uniformDraw([0, 1, 2]);
  factor(n * n);
  return n;
});

viz.auto(y);
~~~~

Without the `factor` statement, each value of the variable `n` has equal probability. Adding the `factor` statements adds `n*n` to the log-score of each value. To get the new probabilities induced by the `factor` statement we compute the normalizing constant given these log-scores. The resulting probability $$P(y=2)$$ is:

$$
P(y=2) = \frac {e^{2 \cdot 2}} { (e^{0 \cdot 0} + e^{1 \cdot 1} + e^{2 \cdot 2}) }
$$

Returning to our implementation as planning-as-inference for maximizing *expected* utility, we use a `factor` statement to implement soft conditioning:

~~~~
var transition = function(state, action){
  var nextStates = ['bad', 'good', 'spectacular'];
  if (action === 'italian'){ 
    return categorical([0.2, 0.6, 0.2], nextStates);
  } else {
    return categorical([0.05, 0.9, 0.05], nextStates);
  };
};

var utility = function(state){
  var table = { bad: -10, good: 6, spectacular: 8 };
  return table[state];
};

var alpha = 1;

var softMaxAgent = function(state){
  return Infer({ method: 'enumerate' }, function(){

    var action = uniformDraw(['french', 'italian']);

    var expectedUtility = function(action){
      return expectation(Infer({ method: 'enumerate' }, function(){
        return utility(transition(state,action));
      }));
    };
    factor(alpha * expectedUtility(action));
    return action;
  })
};

viz.auto(softMaxAgent('default'));
~~~~

The `softMaxAgent` differs in two ways from the `maxEUAgent` above. First, it uses the planning-as-inference idiom. Second, it does not deterministically choose the action with maximal expected utility. Instead, it implements *soft* maximization, selecting actions with a probability that depends on their expected utility. Formally, let the agent's probability of choosing an action be $$C(a;s)$$ for $$a \in A$$ when in state $$s \in S$$. Then the *softmax* decision rule is:

$$
C(a; s) \propto e^{\alpha \mathbb{E}(U(T(s,a))) }
$$

The noise parameter $$\alpha$$ modulates between random choice $$(\alpha=0)$$ and the perfect maximization $$(\alpha = \infty)$$ of the `maxEUAgent`.

Since rational agents will *always* take the best action, why consider softmax agents? If the task is to provide normative advice on how to solve a one-shot decision problem, then "hard" maximization is the way to go. An important goal for this tutorial is to infer the preferences and beliefs of agents from their choices. These agents might not always choose the normatively optimal actions. The softmax agent provides a computationally simple, analytically tractable model of suboptimal choice[^softmax]. This model has been tested empirically on human action selection refp:luce2005individual. Moreover, it has been used extensively in Inverse Reinforcement Learning as a model of human errors refp:kim2014inverse, refp:zheng2014robust. For for this reason, we employ the softmax model throughout this tutorial. When modeling an agent assumed to be optimal, the noise parameter $$\alpha$$ can be set to a large value. <!-- [TODO: Alternatively, agent could output dist.MAP().val instead of dist.] -->

[^softmax]: A softmax agent's choice of action is a differentiable function of their utilities. This differentiability makes possible certain techniques for inferring utilities from choices.

### Moving to complex decision problems

This chapter has introduced some of the core concepts that we will need for this tutorial, including *expected utility*, *(stochastic) transition functions*, *soft conditioning* and *softmax decision making*. These concepts would also appear in standard treatments of rational planning and reinforcement learning refp:russell1995modern. The actual decision problems in this chapter are so trivial that our notation and programs are overkill. The [next chapter](/chapters/3a-mdp.html) introduces *sequential* decisions problems. These problems are more complex and interesting, and will require the machinery we have introduced here. 

<br>

### Footnotes
