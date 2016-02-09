---
layout: chapter
title: "Overview of WebPPL: the language of this tutorial"
description: "WebPPL is a functional subset of Javascript with primitives for sampling from random variables and for Bayesian inference."
is_section: false
---



## Introduction

This chapter introduces the probabilistic programming language WebPPL (pronounced "web people"). The models for agents (and for learning about agents) in this tutorial are all implemented in WebPPL -- so it's an important building block for what follows.

This will be a overview of WebPPL features that are essential to the rest of the tutorial. It will move quickly over the key ideas of probabilistic programming. If you are new to probabilistic programming, you might read a more general introduction (e.g. [here](http://www.pl-enthusiast.net/2014/09/08/probabilistic-programming/) or [here](https://moalquraishi.wordpress.com/2015/03/29/the-state-of-probabilistic-programming/). A detailed [tutorial](https://probmods.org) on Bayesian methods and probabilistic programming, using a language similar to WebPPL, is also good background.

The only requirement to run the code for this tutorial is to a browser (Chrome/Safari). However, to explore the models in more detail, you will want to run WebPPL from the command line. Installation is simple and is explained [here](http://webppl.org).


## WebPPL: a functionally pure subset of Javascript

WebPPL includes a subset of Javascript, and follows the syntax of Javascript for this subset.

This example program uses most of the Javascript syntax that is available in WebPPL:

~~~~
// function definition using JS's Math library

var verboseLog = function(x){
    if (x<=0 || window.isNaN(x)) {
      print("Input " + x + " was not a positive number");
      return null;
    } else {
      return Math.log(x);
    }
};

// Array with numbers, object, Boolean types
var inputs = [1, 1.5, -1, {key:1}, true];

// WebPPL's *map*
map(verboseLog, inputs);   
~~~~

Language features with side effects are not allowed in WebPPL. The commented-out code uses assignment to update a table and produces an error.

~~~~
// Assignment produces an error
// var table = {}
// table.key = 1
// table.key += 1

// Instead do this:
var table = {key: 1};
var updatedTable = {key: table.key + 1};
updateTable;
~~~~

There are no `for` or `while` loops. Instead use higher-order functions like WebPPL's builtin `map`, `filter` and `zip`:

~~~~
var ar = [1,2,3];
// for (var i = 0; i < ar.length; i++){
//   print(ar[i]); }

// Instead of for-loop, use *map* 
map(print,ar);
~~~~

It is possible to use normal Javascript functions (which make internal use of side effects) in WebPPL. See the [online book](http://dippl.org/chapters/02-webppl.html) on the implementation of WebPPL for details (section "Using Javascript Libraries"). 


## WebPPL stochastic primitives

### Sampling from random variables
WebPPL has an array of built-in functions for sampling random variables (i.e. generating random numbers from a particular probability distribution). These will be familiar from scientific computing and probability theory. A full list of functions is in the WebPPL library [source](https://github.com/probmods/webppl/blob/dev/src/header.wppl). 

~~~~

print('Fair coins:', flip(0.5), flip(0.5));
print('Biased coins:', flip(0.9), flip(0.9));

var coinWithSide = function(){
    return categorical( [.49, .49, .02], ['heads', 'tails', 'side']);
};
print(repeat(5,coinWithSide)); // draw i.i.d samples
    

print('Samples from standard Gaussian in 1D:', gaussian(0,1), gaussian(0,1));

print('Samples from 2D Gaussian', multivariateGaussian([0,0],[1,10], multivariateGaussian([0,0],[1,10]));

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
