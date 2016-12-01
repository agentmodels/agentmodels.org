---
layout: chapter
title: "Probabilistic programming in WebPPL"
description: "WebPPL, the language used in this tutorial, is a functional subset of Javascript with primitives for sampling from random variables and for Bayesian inference."
is_section: true
---

## Introduction

This chapter introduces the probabilistic programming language WebPPL (pronounced "web people"). The models for agents in this tutorial are all implemented in WebPPL and so it's important to understand how the language works. 

We begin with a quick overview of probabilistic programming. If you are new to probabilistic programming, you might want to read an informal introduction (e.g. [here](http://www.pl-enthusiast.net/2014/09/08/probabilistic-programming/) or [here](https://moalquraishi.wordpress.com/2015/03/29/the-state-of-probabilistic-programming/)) or a more technical [survey](https://scholar.google.com/scholar?cluster=16211748064980449900&hl=en&as_sdt=0,5). For a practical introduction to both probabilistic programming and Bayesian modeling, we recommend [ProbMods](http://probmods.org). ProbMods is an online tutorial in Church, a language very similar to WebPPL, and its early chapters introduce key ideas in Bayesian generative models that are helpful background.

The only requirement to run the code for this tutorial is a modern browser (e.g. Chrome, Firefox, Safari). If you want to explore the models in detail and to create your own, we recommend running WebPPL from the command line. Installation is simple and is explained [here](http://webppl.org).


## WebPPL: a purely functional subset of Javascript

WebPPL includes a subset of Javascript, and follows the syntax of Javascript for this subset.

This example program uses most of the Javascript syntax that is available in WebPPL:

~~~~
// Function definition using Javascript's `isNaN` and `log` primitives:

var verboseLog = function(x) {
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

WebPPL has a large [library](http://docs.webppl.org/en/master/distributions.html) of primitive probability distributions. Try clicking "Run" repeatedly to get different i.i.d. random samples:

~~~~
print('Fair coins (Bernoulli distribution): ' + [flip(0.5), flip(0.5)]);
print('Biased coins (Bernoulli distribution): ' + [flip(0.9), flip(0.9)]);

var coinWithSide = function(){
  return categorical([.49, .49, .02], ['heads', 'tails', 'side']);
};

print(repeat(5, coinWithSide)); // draw i.i.d samples
~~~~

There are also continuous random variables:

~~~~
print('Two samples from standard Gaussian in 1D: ');
print([gaussian(0, 1), gaussian(0, 1)]);

print('A single sample from a 2D Gaussian: ');
print(multivariateGaussian(Vector([0, 0]), Matrix([[1, 0], [0, 10]])));
~~~~

You can write your own functions to sample from more complex distributions. This example uses recursion to define a sampler for the Geometric distribution:

~~~~
var geometric = function(p) {
  return flip(p) ? 1 + geometric(p) : 1
};

geometric(0.8);
~~~~

What makes WebPPL different from conventional programming languages is its ability to perform *inference* operations using these primitive probability distributions. Distribution objects in WebPPL have two key features:

1. You can draw *random i.i.d. samples* from a distribution using the special function `sample`. That is, you sample $$x \sim P$$ where $$P(x)$$ is the distribution.

2. You can compute the probability (or density) the distribution assigns to a value. That is, to compute $$\log(P(x))$$, you use `dist.score(x)`, where `dist` is the distribution in WebPPL.

The functions above that generate random samples are defined in the WebPPL library in terms of primitive distributions (e.g. `Bernoulli` for `flip` and `Gaussian` for `gaussian`) and the built-in function `sample`:

~~~~
var flip = function(p) {
  var p = (p !== undefined) ? p : 0.5;
  return sample(Bernoulli({ p: p }));
};

var gaussian = function(mu, sigma) {
  return sample(Gaussian({ mu, sigma }));
};

[flip(), gaussian(1, 1)];
~~~~

To create a new distribution, we pass a (potentially stochastic) function with no arguments---a *thunk*---to the function `Infer` that performs *marginalization*. For example, we can use `flip` as an ingredient to construct a Binomial distribution using enumeration:

~~~~
var binomial = function() {
  var a = flip(0.5);
  var b = flip(0.5);
  var c = flip(0.5);
  return a + b + c;
};

var MyBinomial = Infer({ model: binomial });

[sample(MyBinomial), sample(MyBinomial), sample(MyBinomial)];
~~~~

`Infer` is the *inference operator* that computes (or estimates) the marginal probability of each possible output of the function `binomial`. If no explicit inference method is specified, `Infer` defaults to enumerating each possible value of each random variable in the function body.

### Bayesian inference by conditioning

The most important use of inference methods is for Bayesian inference. Here, our task is to *infer* the value of some unknown parameter by observing data that depends on the parameter. For example, if flipping three separate coins produce exactly two Heads, what is the probability that the first coin landed Heads? To solve this in WebPPL, we can use `Infer` to enumerate all values for the random variables `a`, `b` and `c`. We use `condition` to constrain the sum of the variables. The result is a distribution representing the posterior distribution on the first variable `a` having value `true` (i.e. "Heads").  

~~~~
var twoHeads = Infer({
  model() {
    var a = flip(0.5);
    var b = flip(0.5);
    var c = flip(0.5);
    condition(a + b + c === 2);
    return a;
  }
});

print('Probability of first coin being Heads (given exactly two Heads) : ');
print(Math.exp(twoHeads.score(true)));

var moreThanTwoHeads = Infer({  
  model() {
    var a = flip(0.5);
    var b = flip(0.5);
    var c = flip(0.5);
    condition(a + b + c >= 2);
    return a;
  }
});

print('\Probability of first coin being Heads (given at least two Heads): ');
print(Math.exp(moreThanTwoHeads.score(true)));
~~~~

### Codeboxes and Plotting

The codeboxes allow you to modify our examples and to write your own WebPPL code. Code is not shared between boxes. You can use the special function `viz` to plot distributions:

~~~~
var appleOrangeDist = Infer({
  model(){ 
    return flip(0.9) ? "apple" : "orange";
  }
});

viz(appleOrangeDist);
~~~~

~~~~
var fruitTasteDist = Infer({ 
  model(){
    return {
      fruit: categorical([0.3, 0.3, 0.4], ["apple", "banana", "orange"]),
      tasty: flip(0.7)
    };
  }
});

viz(fruitTasteDist);
~~~~

~~~~
var positionDist = Infer({ 
  model() {
    return { 
      X: gaussian(0, 1), 
      Y: gaussian(0, 1)};
  },
  method: 'rejection', 
  samples: 1000
});

viz(positionDist);
~~~~

### Next

In the [next chapter](/chapters/3-agents-as-programs.html), we will implement rational decision-making using inference functions. 
