---
layout: chapter
title: "Overview of WebPPL: the language of this tutorial"
description: "WebPPL is a functional subset of Javascript with primitives for sampling from random variables and for Bayesian inference."
is_section: false
---



## Introduction

This chapter introduces the probabilistic programming language WebPPL (pronounced "web people"). The models for agents (and for learning about agents) in this tutorial are all implemented in WebPPL -- so it's an important building block for what follows.

This will be a overview of WebPPL features that are essential to the rest of the tutorial. It will move quickly over the key ideas of probabilistic programming. If you are new to probabilistic programming, you might read a more general introduction (e.g. [here](http://www.pl-enthusiast.net/2014/09/08/probabilistic-programming/) or [here](https://moalquraishi.wordpress.com/2015/03/29/the-state-of-probabilistic-programming/)). A detailed [tutorial](https://probmods.org) on Bayesian methods and probabilistic programming, using a language similar to WebPPL, is also good background.

The only requirement to run the code for this tutorial is a browser (Chrome/Safari). However, to explore the models in more detail, you will want to run WebPPL from the command line. Installation is simple and is explained [here](http://webppl.org).


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

print("Apply verboseLog to elements in array: "); 
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
updatedTable;
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
WebPPL has an array of built-in functions for sampling random variables (i.e. generating random numbers from a particular probability distribution). These will be familiar from scientific computing and probability theory. A full list of functions is in the WebPPL library [source](https://github.com/probmods/webppl/blob/dev/src/header.wppl). Try clicking the "Run" button repeatedly to get different random samples. 

~~~~

print(['Fair coins:', flip(0.5), flip(0.5)]);
print(['Biased coins:', flip(0.9), flip(0.9)]);

var coinWithSide = function(){
    return categorical( [.49, .49, .02], ['heads', 'tails', 'side']);
};
print(repeat(5, coinWithSide)); // draw i.i.d samples
    

print(['Samples from standard Gaussian in 1D:  ', gaussian(0,1), gaussian(0,1)]);

print(['Sample from 2D Gaussian:  ', 
    multivariateGaussian( [0,0], [[1,0],[0,10]] )]);
    

~~~~

You can write your own functions for sampling random variables . This example uses recursion to define a sampler for the Geometric distription:

~~~~
var geometric = function(p) {
  return flip(p) ? 1 + geometric(p) : 1
};

geometric(0.8);
~~~~

What makes WebPPL different from conventional programming languages is its ability to represent and manipulate *probability distributions*. Elementary Random Primitives (ERPs) are the basic object type that represents distributions. ERP objects have two key features:

1. You can draw *random i.i.d. samples* from an ERP using the special function `sample`. That is, you sample $$x \sim P$$ where $$P(x)$$ is the distribution represented by the ERP. 

2. You can compute the probability (or density) the distribution assigns to a value. That is, to compute $$\log(P(x))$$, you use `erp.score([], x)`, where `erp` is the ERP in WebPPL. 

The functions above that generate random samples are defined in the WebPPL library in terms of built-in ERPs (e.g. `bernoulliERP` for `flip` and `gaussianERP` for `gaussian`) and the built-in function `sample`.

To create a new ERP, we pass a *thunk* (a function with no arguments) with a random output, to a function that performs *marginalization*. For example, we can use `bernoulliERP` as an ingredient to construct a Binomial distribution with the marginalization function `Enumerate`:

~~~~
var binomial = function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  return a + b + c;
};

var binomialERP = Enumerate(binomial);

[sample(binomialERP), sample(binomialERP), sample(binomialERP)];
~~~~

The function `Enumerate` is an *inference function* that computes the marginal probability of each possible output of the function `binomial` by enumerating each possible value of the random variables (`a`, `b` and `c`) in the function body.

### Bayesian inference by conditioning
The most important use of `Enumerate` is for Bayesian inference. Our task is to *infer* the value of some unknown parameter by observing data that depends on the parameter. For example, if flipping three separate coins produce exactly two Heads, what is the probability that the first coin landed Heads? To solve this in WebPPL, we use `Enumerate` to enumerate all values for the random variables `a`, `b` and `c`. We use `condition` to constrain the sum of the variables. The result is an ERP representing the posterior distribution on the first variable `a` having value `true` (i.e. "Heads").  

~~~~
var twoHeads = Enumerate(function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  condition( a + b + c == 2 );
  return a;
});

print('Probability first coin being Heads (given exactly two Heads) : ');
print(Math.exp(twoHeads.score([],true)));

var moreThanTwoHeads = Enumerate(function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  condition( a + b + c >= 2 );
  return a;
});

print('\Probability on first coin being Heads (given at least two Heads): ');
print(Math.exp(moreThanTwoHeads.score([],true)));
~~~~

### Codeboxes and Plotting
You can use the code boxes to modify our examples or to write your own WebPPL code. Code is not shared between boxes. You can use the special function `viz.print` to plot ERPs:

~~~~
var discrete_erp = Enumerate(function(){ 
        return flip(0.9) ? "apple" : "orange";
    });
viz.print(discrete_erp);
~~~~

~~~~
var discrete_erp_2d = Enumerate(function(){
  return {
    fruit: categorical([0.3, 0.3, 0.4], ["apple", "banana", "orange"]),
    boolean: flip(0.7)
  };
  });
viz.print(discrete_erp_2d);
~~~~

~~~~
var continuous_erp_2d = ParticleFilter(function(){
  return { X: gaussian(0, 1), Y: gaussian(0, 1)};
  }, 1000);
viz.print(continuous_erp_2d);
~~~~

### Next

In the next [chapter](/chapters/03-one-shot-planning.html), we use inference functions to implementing rational decision making. 


