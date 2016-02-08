---
layout: chapter
title: Building agent models in WebPPL
description: "Introduction to WebPPL: a functional subset of Javascript, with primitives for sampling from random variables and for Bayesian inference "
is_section: true
---


## Introduction

This chapter introduces the probabilistic programming language WebPPL (pronounced "web people") that we use to model agents throughout this tutorial. We give a brief overview of the features that are essential to this tutorial. If you have never encountered probabilistic programming before, you might benefit from reading some introductory material. There are short articles [here](http://plenthusiast) and [here](http://mohammed) that provide a general overview. There is an interactive tutorial covering probabilistic programming and Bayesian inference at [probmods](https://probmods.org), which uses a language very similar to WebPPL. If you have some background in programming languages, there is a [tutorial](https://dippl.org) on how to implement WebPPL (which will also give a sense of how the language works).

Most of the code examples we provide will run in your browser. WebPPL can also be installed locally and run from the command line --- see [here](https://webppl.org).

## WebPPL: a functionally pure subset of Javascript

WebPPL includes a subset of Javascript, and follows the syntax of Javascript for this subset. (Since we only use a limited subset of Javascript, you will only need a basic knowledge of Javascript to use WebPPL). 

This program uses most of the available JS syntax:

~~~~
var verboseLog = function(x) {
    if (x<=0 || window.isNaN(x)) {
      print("Input " + x + " was not a positive number");
      return null;
    } else {
      return Math.log(x);
    }
};

var inputs = [1, 2, -1, {key:1}, true];
map(verboseLog, inputs);
~~~~

Language features with side effects are not allowed in WebPPL. The commented-out code uses assignment to update a table and produces an error.

~~~~
// var table = {}
// table.key = 1
// table.key += 1

// Instead do this:
var table = {key: 1};
var updatedTable = {key: table.key + 1};
~~~~

There are no `for` and `while` loops. Instead use higher-order functions like `map`:

~~~~
var ar = [1,2,3];
// for (var i = 0; i < ar.length; i++){
//   print('array element:' + ar[i])}

// Instead of for-loop, use *map* (built-in for WebPPL)
map(function(i){print('array element: ' + i)}, ar);
~~~~

Normal Javascript functions can be called from WebPPL (with some restrictions). See here [Dippl chapter 1] for details. 


## WebPPL stochastic primitives

### Sampling from random variables
WebPPL has an array of built-in functions for sampling random variables (i.e. generating random numbers from a particular probability distribution). These will be familiar from other scientific/numeric computing libraries.

~~~~
var fairCoinFlip = flip(0.5);
var biasedCoinFlip = flip(0.6);
var integerLess6 = uniformDraw([1,2,3,4,5]);
var coinWithSide = categorical( [.49, .49, .02], ['heads', 'tails', 'side']);

var gaussianDraw = gaussian(0,1);

[fairCoinFlip, biasedCoinFlip, integerLess6, coinWithSide, gaussianDraw];
~~~~

Additional functions for sampling random variables can be defined. This example uses recursion to define a sampler for the Geometric distription with parameter `p`:

~~~~
var geometric = function(p) {
  return flip(p) ? 1 + geometric(p) : 1
};

geometric(0.1);
~~~~

What makes WebPPL different from conventional programming languages is its ability to represent and manipulate *probability distributions*. Elementary Random Primitives (ERPs) are the basic object type that represents distributions. ERPs allow you to sample values from a distribution. But they also allow you to compute the log-probability of a possible sampled value.

The functions above that generate random samples are defined in the WebPPL library in terms of built-in ERPs (e.g. `bernoulliERP` for `flip` and `gaussianERP` for `gaussian`) and the built-in function `sample`.

To create a new ERP, we pass a *thunk* (a function with no arguments) which has a random output, to a *marginalization* or *inference* function. For example, we can use `bernoulliERP` to construct a Binomial distribution:

~~~~
var binomial = function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  return a + b + c;
}

var binomialERP = Enumerate(binomial);

viz.print(binomialERP);

[sample(binomialERP), sample(binomialERP), sample(binomialERP)];
~~~~

The function `Enumerate` is an *inference function* that computes the marginal probability of each possible output of the function `binomial` by enumerating (using a standard search algorithm) each possible value of the random variables (`a`, `b` and `c`) in the function body. Using `sample` we generate random binomial samples.

### Bayesian inference by conditioning
The most important use of inference functions like `Enumerate` is for inference. Suppose a function produces random outputs. If there are some random variables in the body of the function, we can ask: what is the likely value of the random variables given the observed output of the function? For example, if three fair coins produce exactly two Heads, what is the probability that the first coin landed Heads? [Maybe use an example that isn't so intractable]. 

~~~~
var twoHeads = Enumerate(function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  condition( a + b + c == 2 );
  return a;
});

viz.print(twoHeads);

var moreThanTwoHeads = Enumerate(function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  condition( a + b + c >= 2 );
  return a;
});

viz.print(moreThanTwoHeads);
~~~~

In the next chapter, we use inference functions to implementing rational decision making.

Next chapter: [Modeling simple decision problems](/chapters/03-one-shot-planning.html)
