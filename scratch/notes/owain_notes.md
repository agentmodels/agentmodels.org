
# Chapter 2: Probabilistic programming in Webppl

## Introduction
This chapter introduces the probabilistic programming language Webppl (pronounced "web people") that we use to model agents throughout this tutorial. We give a brief overview of the features that are essential to this tutorial. If you have never encountered probabilistic programming before, you might benefit from reading some introductory material. There short articles [here](plenthusiast) and [here](mohammed) that provide a general overview. There is an interactive tutorial covering probabilistic programming and Bayesian inference at [probmods](https://probmods.org), which uses a language very similar to Webppl. If you have some background in programming languages, there is a [tutorial](https://dippl.org) on how to implement Webppl (which will also give a sense of how the language works).

Most of the code examples we provide will run in your browser (ideally Chrome/Safari). Webppl can also be installed locally and run from the command line --- see [here](https://webppl.org).


## Webppl: a functionally pure subset of Javascript
Webppl includes a subset of Javascript, and follows the syntax of Javascript for this subset. (Since we only use a limited subset of Javascript, you will only need a basic knowledge of Javascript to use Webppl). 

This program uses most of the available JS syntax:

~~~~
// Output message when input to log is not positive
var verboseLog = function(x) {
    var message = "Message: input to log was not a positive number"
    return (x<0 || x==0) ? message : Math.log(x)  // single-line conditional syntax
}

// Reminder that Javascript will coerce non-Numeric types into numbers
[verboseLog(1), verboseLog(-1), verboseLog({key:1}), verboseLog(true)] 

~~~~

Language features with side effects are not allowed in Webppl. The following examples show to achieve the same behavior in Webppl:

~~~~
// assignment to a variable or attribute is not allowed:
// var table = {}
// table.key = 1
// table.key += 1

// Instead:
var table = {key: 1}
var updatedTable = {key: table.key + 1}

// *for* and *while* loops are not allowed
var ar = [1,2,3]
// for (var i = 0; i < ar.length; i++){
//   console.log('array element:', ar[i])}

// Instead, use a function in standard Webppl library "map"
map( function(i){console.log('array element:')}, ar); 
~~~~

## Webppl stochastic primitives

### Sampling from random variables
Webppl has an array of built-in functions for sampling random variables (i.e. generating random numbers from a particular probability distribution). These will be familiar from other scientific/numeric computing libraries.

~~~~
var fairCoinflip = flip(0.5)
var biasedCoinflip = flip(0.6)
var integerLess6 = uniformDraw([1,2,3,4,5])
var coinWithSide = categorical( [.49, .49, .02], ['heads', 'tails', 'side'])

var gaussianDraw = gaussian(0,1)
~~~~

Additional functions for sampling random variables can be defined. This example uses recursion to define a sampler for the Geometric distription with parameter `p`: 
~~~
var geometric = function(p) {
  return flip(p)?1+geometric(p):1
}

geometric(0.5)
~~~

What makes Webppl different from conventional programming languages is its ability to represent and manipulate *probability distributions*. Elementary Random Primitives (ERPs) are the basic object type that represents distributions. ERPs allow you to sample values from a distribution. But they also allow you to compute the log-probability of a possible sampled value.

The functions above that generate random samples are defined in the Webppl library in terms of built-in ERPs (e.g. `bernoulliERP` for `flip` and `gaussianERP` for `gaussian`) and the built-in function `sample`.

To create a new ERP, we pass a *thunk* (a function with no arguments) which has a random output, to a *marginalization* or *inference* function. For example, we can use `bernoulliERP` to construct a Binomial distribution:

~~~
var binomial = function(){
  var a = flip(0.5)
  var b = flip(0.5)
  var c = flip(0.5)
  return a + b + c}

var binomialERP = Enumerate(binomial)

print(binomialERP)
[sample(binomialERP), sample(binomialERP), sample(binomialERP)]
~~~

The function `Enumerate` is an *inference function* that computes the marginal probability of each possible output of the function `binomial` by enumerating (using a standard search algorithm) each possible value of the random variables (`a`, `b` and `c`) in the function body. Using `sample` we generate random binomial samples.

### Bayesian inference by conditioning
The most important use of inference functions like `Enumerate` is for *inference*. Suppose a function produces random outputs. If there are some random variables in the body of the function, we can ask: what is the likely value of the random variables given the observed output of the function? For example, if three fair coins produce exactly two Heads, what is the probability that the first coin landed Heads? [Maybe use an example that isn't so intractable]. 

~~~~
var twoHeads = function(){
  var a = flip(0.5)
  var b = flip(0.5)
  var c = flip(0.5)
  condition( a + b + c == 2 )
  return a}

sample(firstCoin) // samples of value of first coin, conditioned on total being 2 Heads

var moreThanTwoHeads = function(){
  var a = flip(0.5)
  var b = flip(0.5)
  var c = flip(0.5)t
  condition( a + b + c >= 2 )
  return a}

sample(moreThanTwoHeads)
~~~~

In the next chapter, we use inference functions to implementing rational decision making.

Next chapter: [Modeling simple decision problems](/chapters/03-decisions.html)


# Chapter 3: Modeling simple decision problems

1. take option with maxU consequences (deterministic). planning as inference formalism. 
2. take option with max exptU consequences. need expectation. 

PLAN:

//MDP and POMDP models

// stuff written

// 

// plans for this week

Go through details of getting code working. 
Some plans for Tuesday: need some concrete tasks. 

Continuing on with MDP and POMDP agents. That's where we need Gridworld. So one thing for Tuesday could be writing spec for Gridworld, including looking at standard examples. With spec for Gridworld, we can also look at concrete instances of the complexity question. 

MDP agent: simplify by removing simulate/expU distinction. (MDP can be made simpler but will have less continuity with biases version). 

POMDP agent: investigate caching issue. Look at complexity for multi-arm bandits. A task is to implement stochastic bandits in the same framework.

We'd like to do the restaurant example in a Gridworld (many movements ruled out) and with 


## naming:     state={ observedState:'A', latentState:{B:1, A:1, etc.} }
## don't call the agent's beliefState a state. call it posterior on latent states




## discuss VOI somewhere in POMDP section


## MDP and POMDP: implement Andreas vocab change ASAP

## MDP and POMDP: if rewards are stochastic, natural to record how well the agent actually did (fix bandit case). 




## Speculative tasks
- Do abbeel and ng algorithms for app learning in webppl/JS
- Max ent IRL
- Implement Q-learning in webppl. To what degree can it be done in a similar style to current MDP agent?


## Gridworld notes

## Spec for Gridworld and Visualization

### Implementation of Gridworld
My aim is to have a Gridworld implementation that fits the standard examples used in RL/MDP (Russell-Norvig 4x3 example, cliff example, and so on). It should also fit the Restaurant Choice examples from our papers.

I've implemented a quick version of the standard Gridworld in ppl_files/models/bookMDP.wppl, after consulting the Berkeley "Pac-Man" python package and the Russell-Norvig AIMA(3e) text. I'll summarize the key features:

#### Utility when leaving states
Pairs of [state,action], i.e. your current state and the action you take in that state, are the argument for the transition function T and the utility function U. In standard Gridworld states have utilities and there is a generic action cost. So U(s,a) = U*(s) - .15 (for example).

On our model, the agent gets utility on *leaving* a state. Assume for illustration that actions don't matter for utility. Suppose the agent has only a single action and starts in state s0. Then they will receive U(s0), no matter what action they take. Note that wherever they would go after s0 doesn't make any difference to their utility. (Also note that we learn nothing about the agent their choice of action). 

This is the model (utility when leaving) that the Berkeley-Pacman group follows. AIMA does it differently. If we have terminal states as in standard Gridworld (where the agent can get a reward from state before the game ends), we need to set up the terminal states such that you get utility. Berkeley does this by having a special "exit" action that you take in terminals. 


#### Cells either open or blocked
There are *open* and *blocked* grid cells. You start in an open cell and if you walk towards a blocked cell you stay put. Likewise, if you walk off the grid (i.e. "against a wall") you stay put.

#### Terminal nodes
Standard MDPs can have unbounded horizon. We have focused on finite horizon. For unbounded horizon, you need *terminal* nodes to end the game. Should we have terminal nodes? In the Restaurant Example, we used terminal nodes. We could simulate having terminal nodes by adding an extra invisible "sink" state that you must go to after visiting what would be the terminal state. (Not having terminals means that we don't need checks in the code for "isTerminal"). On the other hand, having real terminals brings this closer to the standard MDP setting (and might in some cases lead to runtime savings).

#### Stochastic transitions
Standard examples have a noise model where you go LEFT or RIGHT with probability .1 when you choose UP (and so on). This is easy enough to implement. (I don't think it complicates the current MDP code to have stochastic transitions -- but I need to verify this). 

#### Stochastic rewards
Bandit problems have stochastic rewards. It seems fairly easy to change the code to do this for Gridworld. 


### Visualization of Gridworld

Here are the desired features for visualization:

1. Shows blocked/open cells (looking like a path or road map). Highlights terminals/restaurants and shows the utility of cells with non-zero utility.

2. Show the agent's path over time. The simplest way is an animation. However, I think showing the path in a static image is very useful. The path could be a series of arrows (direction of agent from each cell). But best would be a line joining midpoints of successive cells, with occasional arrows showing overall direction. 

3. Label the agent's policy and the EU of each state (a discrete heat map). For MDP and POMDP this is time consistent. For time-inconsistent agents, it will be awesome to show how this heatmap varies over time (as the agent re-computes the EU). 

4. For noisy agents, could show a heatmap of the agent's time spent in each cell. 

5. Interactive: user can manipulate the utilities or other features of the MDP (as in Karpathy's JS DP example) or user can drive the agent and so influence the IRL that is done.


Here are some thoughts on what kind of data structure to provide to the visualization module. The basic idea is that each cell has a set of fixed features {utility:5, blocked:false, terminal:true} and possibly also {agentPolicy:'r', agentQValue:{u:4.6, d:2.2, l:4, r:1}}. Then you have a sequence of agent actions of form [ cell1, cell2, ... ]. You create paths by adding horizontal or vertical lines to cells, based on the diffs of the sequence. These could be extra attributes of cells, or could be represented in a separate layer. (Or whatever the standard solution is here). 

In Webppl, you have a function for producing all gridcells (e.g. by enumerating) and then functions from states to their various features. Features like `isBlocked`, `isTerminal`, and `utility(state)` are part of the `params` for Gridworld MDPs. The agent-dependent features (e.g. the agent's action in a cell, the agent's EU for a cell) have to be computed using the `agent` function. You compute something like zip(allCells, map( getAllFeatures, allCells)) and send this to JS for visualization. 





some datastructure that is too be displayed. includes U, blocked, terminal, whether agent present. for videos we'll have sequence of agent positions and add them to this. to draw trajectory, you want to draw a trail through the grid. so you draw lines joining centers of squares in a sequence. then there's showing a policy and q values

so basic info:
{u:5, blocked:false, terminal:true}

for policy and Q values:
{policy:'r', V:4.5, Q:{u:4.6, d:2.2, l:4, r:1} }

to draw paths, you need the whole sequence. you can then annotate cells with lines. 

TODO:
- add terminals to MDP model
