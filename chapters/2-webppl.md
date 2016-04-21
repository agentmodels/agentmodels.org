---
layout: chapter
title: "Probabilistic programming in WebPPL"
description: "WebPPL, the language used in this tutorial, is a functional subset of Javascript with primitives for sampling from random variables and for Bayesian inference."
is_section: true
---

## Introduction

This chapter introduces the probabilistic programming language WebPPL (pronounced "web people"). The models for agents in this tutorial are all implemented in WebPPL and so it's important to understand how the language works. 

We begin with a quick overview of probabilistic programming. If you are new to probabilistic programming, you might want to read an informal introduction (e.g. [here](http://www.pl-enthusiast.net/2014/09/08/probabilistic-programming/) or [here](https://moalquraishi.wordpress.com/2015/03/29/the-state-of-probabilistic-programming/)) or a more technical [survey](https://scholar.google.com/scholar?cluster=16211748064980449900&hl=en&as_sdt=0,5). For a practical introduction to both probabilistic programming and Bayesian modeling, we recommend [Prob Mods](http://probmods.org). Prob Mods is an online tutorial in Church, a language very similar to WebPPL, and its early chapters introduce key ideas in Bayesian generative models that are helpful background.

The only requirement to run the code for this tutorial is a modern browser (e.g. Chrome, Firefox, Safari). If you want to explore the models in detail and to create your own, we recommend running WebPPL from the command line. Installation is simple and is explained [here](http://webppl.org).


## WebPPL: a purely functional subset of Javascript

WebPPL includes a subset of Javascript, and follows the syntax of Javascript for this subset.

This example program uses most of the Javascript syntax that is available in WebPPL:

~~~~
// Function definition using Javascript's `isNaN` and `log` primitives:

var verboseLog = function(x){
  if (x<=0 || _.isNaN(x)) {
    print("Input " + x + " was not a positive number");
    return null;
  } else {
    return Math.log(x);
  }
};

// Array with numbers, object, Boolean types
var inputs = [1, 1.5, -1, {key: 1}, true];

print("Apply verboseLog to elements in array: "); 
map(verboseLog, inputs);
~~~~

Language features with side effects are not allowed in WebPPL. The code that has been commented out uses assignment to update a table. This produces an error in WebPPL.

~~~~
// Don't do this:

// var table = {};
// table.key = 1;
// table.key = table.key + 1;
// => Assignment is allowed only to fields of globalStore.


// Instead do this:

var table = {key: 1};
var updatedTable = {key: table.key + 1};
print(updatedTable);

// Or use the library function *update*:

var secondUpdatedTable = update(table, {key:10})
print(secondUpdatedTable);
~~~~

There are no `for` or `while` loops. Instead, use higher-order functions like WebPPL's built-in `map`, `filter` and `zip`:

~~~~
var ar = [1,2,3];

// Don't do this:

// for (var i = 0; i < ar.length; i++){
//   print(ar[i]); 
// }


// Instead of for-loop, use `map`:
map(print, ar);
~~~~

It is possible to use normal Javascript functions (which make *internal* use of side effects) in WebPPL. See the [online book](http://dippl.org/chapters/02-webppl.html) on the implementation of WebPPL for details (section "Using Javascript Libraries"). 


## WebPPL stochastic primitives

### Sampling from random variables

WebPPL has a number of built-in functions for sampling from random variables from different distributions. Many of these are found in scientific computing libraries. A full list of functions is in the WebPPL library [source](https://github.com/probmods/webppl/blob/dev/src/header.wppl). Try clicking the "Run" button repeatedly to get different i.i.d. random samples:

~~~~
print('Fair coins: ' + [flip(0.5), flip(0.5)]);
print('Biased coins: ' + [flip(0.9), flip(0.9)]);

var coinWithSide = function(){
  return categorical([.49, .49, .02], ['heads', 'tails', 'side']);
};

print(repeat(5, coinWithSide)); // draw i.i.d samples
~~~~

There are also continuous random variables:

~~~~
print('Two samples from standard Gaussian in 1D: ' +
      [gaussian(0,1), gaussian(0,1)]);

print('A single sample from a 2D Gaussian: ' +
      multivariateGaussian([0,0], [[1,0],[0,10]]));
~~~~

You can write your own functions to sample from more complex distributions. This example uses recursion to define a sampler for the Geometric distribution:

~~~~
var geometric = function(p) {
  return flip(p) ? 1 + geometric(p) : 1
};

geometric(0.8);
~~~~

What makes WebPPL different from conventional programming languages is its ability to represent and manipulate *probability distributions*. Elementary Random Primitives (ERPs) are the basic object type that represents distributions. ERP objects have two key features:

1. You can draw *random i.i.d. samples* from an ERP using the special function `sample`. That is, you sample $$x \sim P$$ where $$P(x)$$ is the distribution represented by the ERP. 

2. You can compute the probability (or density) the distribution assigns to a value. That is, to compute $$\log(P(x))$$, you use `erp.score([], x)`, where `erp` is the ERP in WebPPL.

The functions above that generate random samples are defined in the WebPPL library in terms of built-in ERPs (e.g. `bernoulliERP` for `flip` and `gaussianERP` for `gaussian`) and the built-in function `sample`:

~~~~
var flip = function(theta) {
  var theta = (theta !== undefined) ? theta : 0.5;
  return sample(bernoulliERP, [theta]);
};

var gaussian = function(mu, sigma) {
  return sample(gaussianERP, [mu, sigma]);
};

[flip(), gaussian(1, 1)];
~~~~

To create a new ERP, we pass a (potentially stochastic) function with no arguments---a *thunk*---to a function that performs *marginalization*. For example, we can use `flip` as an ingredient to construct a Binomial distribution using the marginalization function `Enumerate`:

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

The most important use of inference functions such as `Enumerate` is for Bayesian inference. Here, our task is to *infer* the value of some unknown parameter by observing data that depends on the parameter. For example, if flipping three separate coins produce exactly two Heads, what is the probability that the first coin landed Heads? To solve this in WebPPL, we can use `Enumerate` to enumerate all values for the random variables `a`, `b` and `c`. We use `condition` to constrain the sum of the variables. The result is an ERP representing the posterior distribution on the first variable `a` having value `true` (i.e. "Heads").  

~~~~
var twoHeads = Enumerate(function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  condition(a + b + c === 2);
  return a;
});

print('Probability of first coin being Heads (given exactly two Heads) : ');
print(Math.exp(twoHeads.score([], true)));

var moreThanTwoHeads = Enumerate(function(){
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  condition(a + b + c >= 2);
  return a;
});

print('\Probability of first coin being Heads (given at least two Heads): ');
print(Math.exp(moreThanTwoHeads.score([], true)));
~~~~

### Codeboxes and Plotting

The codeboxes allow you to modify our examples and to write your own WebPPL code. Code is not shared between boxes. You can use the special function `viz.auto` to plot ERPs:

~~~~
var appleOrangeERP = Enumerate(function(){ 
  return flip(0.9) ? "apple" : "orange";
});

viz.auto(appleOrangeERP);
~~~~

~~~~
var fruitTasteERP = Enumerate(function(){
  return {
    fruit: categorical([0.3, 0.3, 0.4], ["apple", "banana", "orange"]),
    tasty: flip(0.7)
  };
});

viz.auto(fruitTasteERP);
~~~~

~~~~
var positionERP = Rejection(function(){
  return { 
    X: gaussian(0, 1), 
    Y: gaussian(0, 1)};
}, 1000);

viz.auto(positionERP);
~~~~

### Next

In the [next chapter](/chapters/3-agents-as-programs.html), we will implement rational decision-making using inference functions. 
