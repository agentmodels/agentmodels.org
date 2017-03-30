---
layout: chapter
title: Modeling Agents & Reinforcement Learning with Probabilistic Programming
hidden: true
---

## Intro

### Motivation

Why probabilistic programming?
- **ML:** predictions based on prior assumptions and data
- **Deep Learning:** lots of data + very weak assumptions
- **Rule-based systems:** strong assumptions + little data
- **Probabilistic programming:** a flexible middle ground

Why model agents?
- Build **artificial agents** to automate decision-making
    - Example: stock trading
- **Model humans** to build helpful ML systems
    - Examples: recommendation systems, dialog systems

### Preview

What to get out of this talk:
- Intuition for programming in a PPL
- Core PPL concepts
- Why are PPLs uniquely suited for modeling agents?
- Idioms for writing agents as PPs
- How do RL and PP relate?

What not to expect:
- Lots of applications
- Production-ready systems

## Probabilistic programming basics

### Our language: WebPPL

Try it at [webppl.org](http://webppl.org)

### A functional subset of JavaScript

Why JS?
- Fast
- Rich ecosystem
- Actually a nice language underneath all the cruft
- Runs locally via node.js, but also in browser:
    - [SmartPages](https://stuhlmueller.org/smartpages/)
    - [Image inference viz](http://dippl.org/examples/vision.html)
    - [Spaceships](http://dritchie.github.io/web-procmod/)
    - [Agent viz](http://agentmodels.org/chapters/3b-mdp-gridworld.html#hiking-in-gridworld)

~~~~
var xs = [1, 2, 3, 4];

var square = function(x) {
  return x * x;
};

map(square, xs);
~~~~

### Distributions and sampling

Docs: [distributions](http://docs.webppl.org/en/dev/distributions.html)

#### Discrete distributions

Examples: `Bernoulli`, `Categorical`

Sampling helpers: `flip`, `categorical`

~~~~
var dist = Bernoulli({ p: 0.3 });

var flip = function(p) {
  return sample(Bernoulli({ p }));
}

flip(.3)
~~~~

#### Continuous distributions

Examples: `Gaussian`, `Beta`

~~~~
var dist = Gaussian({ 
  mu: 1,
  sigma: 0.5
});

viz(repeat(1000, function() { return sample(dist); }));
~~~~

#### Building complex distributions out of simple parts

Example: geometric distribution

~~~~
var geometric = function(p) {
  if (flip(p)) {
    return 0;
  } else {
    return 1 + geometric(p);
  }
};

viz(repeat(100, function() { return geometric(.5); }));
~~~~

### Inference

#### Reifying distributions

`Infer` reifies the geometric distribution so that we can compute probabilities:

~~~~
var geometric = function(p) {
  if (flip(p)) {
    return 0;
  } else {
    return 1 + geometric(p);
  }
};

var model = function() {
  return geometric(.5);
};

var dist = Infer({
  model,
  maxExecutions: 100
});

viz(dist);

Math.exp(dist.score(3))
~~~~

#### Computing conditional distributions

Example: inferring the weight of a geometric distribution

~~~~
var geometric = function(p) {
  if (flip(p)) {
    return 0;
  } else {
    return 1 + geometric(p);
  }
}

var model = function() {
  var u = uniform(0, 1);
  var x = geometric(u);
  condition(x < 4);
  return u;
}

var dist = Infer({
  model,
  method: 'rejection',
  samples: 1000
})

dist
~~~~

#### Technical note: three ways to condition

~~~~
var model = function() {
  var p = flip(.5) ? 0.5 : 1;
  var coin = Bernoulli({ p });

  var x = sample(coin);
  condition(x === true);
  
//  observe(coin, true);
  
//  factor(coin.score(true));
  
  return { p };
}

viz.table(Infer({ model }));
~~~~

#### A slightly less toy example: regression

Docs: [inference algorithms](http://docs.webppl.org/en/master/inference/methods.html)

~~~~
var xs = [1, 2, 3, 4, 5];
var ys = [2, 4, 6, 8, 10];

var model = function() {
  var slope = gaussian(0, 10);
  var offset = gaussian(0, 10);
  var f = function(x) {
    var y = slope * x + offset;
    return Gaussian({ mu: y, sigma: .1 })
  };
  map2(function(x, y){
    observe(f(x), y)
  }, xs, ys)
  return { slope, offset };
}

Infer({
  model,
  method: 'MCMC',
  kernel: {HMC: {steps: 10, stepSize: .01}},
  samples: 2000,
})
~~~~

## Agents as probabilistic programs

### Deterministic choices

~~~~
var actions = ['italian', 'french'];

var outcome = function(action) {
  if (action === 'italian') {
    return 'pizza';
  } else {
    return 'steak frites';
  }
};

var actionDist = Infer({ 
  model() {
    var action = uniformDraw(actions);
    condition(outcome(action) === 'pizza');
    return action;
  }
});

actionDist
~~~~

### Expected utility

~~~~
var actions = ['italian', 'french'];

var transition = function(state, action) {
  var nextStates = ['bad', 'good', 'spectacular'];
  var nextProbs = ((action === 'italian') ? 
                   [0.2, 0.6, 0.2] : 
                   [0.05, 0.9, 0.05]);
  return categorical(nextProbs, nextStates);
};

var utility = function(state) {
  var table = { 
    bad: -10, 
    good: 6, 
    spectacular: 8 
  };
  return table[state];
};

var expectedUtility = function(action) {
  var utilityDist = Infer({
    model: function() {
      var nextState = transition('initialState', action);
      var u = utility(nextState);
      return u;
    }
  });
  return expectation(utilityDist);
};

map(expectedUtility, actions);
~~~~

### Softmax-optimal decision-making

~~~~
var actions = ['italian', 'french'];

var transition = function(state, action) {
  var nextStates = ['bad', 'good', 'spectacular'];
  var nextProbs = ((action === 'italian') ? 
                   [0.2, 0.6, 0.2] : 
                   [0.05, 0.9, 0.05]);
  return categorical(nextProbs, nextStates);
};

var utility = function(state) {
  var table = { 
    bad: -10, 
    good: 6, 
    spectacular: 8 
  };
  return table[state];
};

var alpha = 1;

var agent = function(state) {
  return Infer({ 
    model() {

      var action = uniformDraw(actions);
      
      var expectedUtility = function(action) {
        var utilityDist = Infer({
          model: function() {
            var nextState = transition('initialState', action);
            var u = utility(nextState);
            return u;
          }
        });
        return expectation(utilityDist);
      };
      
      var eu = expectedUtility(action);
      
      factor(eu);
      
      return action;
      
    }
  });
};

agent('initialState');
~~~~

## Sequential decision problems

- [Restaurant Gridworld](http://agentmodels.org/chapters/3a-mdp.html) (1, last)
- Structure of expected utility recursion
- Dynamic programming


~~~~
var act = function(state) {
  return Infer({ model() {
    var action = uniformDraw(stateToActions(state));
    var eu = expectedUtility(state, action);
    factor(eu);
    return action;
  }});
};

var expectedUtility = function(state, action){
  var u = utility(state, action);
  if (isTerminal(state)){
    return u; 
  } else {
    return u + expectation(Infer({ model() {
      var nextState = transition(state, action);
      var nextAction = sample(act(nextState));
      return expectedUtility(nextState, nextAction);
    }}));
  }
};
~~~~

- [Hiking Gridworld](http://agentmodels.org/chapters/3b-mdp-gridworld.html) (1, 2, 3, last)
- Expected state-action utilities (Q values)
- [Temporal inconsistency](http://agentmodels.org/chapters/5b-time-inconsistency.html) in Restaurant Gridworld
    

## Reasoning about agents

- [Learning about preferences from observations](http://agentmodels.org/chapters/4-reasoning-about-agents.html) (1 & 2)

## Multi-agent models

### A simple example: Coordination games

~~~~
var locationPrior = function() {
  if (flip(.55)) {
    return 'popular-bar';
  } else {
    return 'unpopular-bar';
  }
}

var alice = dp.cache(function(depth) {
  return Infer({ model() {
    var myLocation = locationPrior();
    var bobLocation = sample(bob(depth - 1));
    condition(myLocation === bobLocation);
    return myLocation;
  }});
});

var bob = dp.cache(function(depth) {
  return Infer({ model() {
    var myLocation = locationPrior();
    if (depth === 0) {
      return myLocation;
    } else {
      var aliceLocation = sample(alice(depth));
      condition(myLocation === aliceLocation);
      return myLocation;
    }
  }});
});

alice(5)
~~~~

### Other examples

- [Game playing: tic-tac-toe](http://agentmodels.org/chapters/7-multi-agent.html)
- [Language understanding](http://agentmodels.org/chapters/7-multi-agent.html)

## Reinforcement learning

### Algorithms vs Models

- Models: encode world knowledge
    - PPLs suited for expressing models
- Algorithms: encode mechanisms (for inference, optimization)
    - RL is mostly about algorithms
- But some algorithms can be expressed using PPL components

### Inference vs. Optimization

~~~~
var k = 3;  // number of heads
var n = 10; // number of coin flips

var model = function() {
  var p = sample(Uniform({ a: 0, b: 1}));
  var dist = Binomial({ p, n });
  observe(dist, k);
  return p;
};

var dist = Infer({ 
  model,
  method: ‘MCMC',
  samples: 100000,
  burn: 1000
});

expectation(dist);
~~~~

~~~~
var k = 3;  // number of heads
var n = 10; // number of coin flips

var model = function() {
  var p = Math.sigmoid(modelParam({ name: 'p' }));
  var dist = Binomial({ p, n });
  observe(dist, k);
  return p;
};

Optimize({
  model,
  steps: 1000,
  optMethod: { sgd: { stepSize: 0.01 }}
});

Math.sigmoid(getParams().p);
~~~~



### Policy Gradient

~~~~
///fold:
var numArms = 10;

var meanRewards = map(
  function(i) {
    if ((i === 7) || (i === 3)) {
      return 5;
    } else {
      return 0;
    }
  },
  _.range(numArms));

var blackBox = function(action) {
  var mu = meanRewards[action];
  var u = Gaussian({ mu, sigma: 0.01 }).sample();
  return u;
};
///

// actions: [0, 1, 2, ..., 9]

// blackBox: action -> utility

var agent = function() {
  var ps = softmax(modelParam({ dims: [numArms, 1], name: 'ps' }));
  var action = sample(Discrete({ ps }));
  var utility = blackBox(action);
  factor(utility);
  return action;
};


Optimize({ model: agent, steps: 10000 });

var params = getParams();
viz.bar(
  _.range(10),
  _.flatten(softmax(params.ps[0]).toArray()));
~~~~

## Conclusion

What to get out of this talk, revisited:

- **Intuition for programming in a PPL**
- **Core PPL concepts**
    - Distributions & samplers
    - Inference turns samplers into distributions
    - `sample` turns distributions into samples
    - Optimization fits free parameters
- **Idioms for writing agents as probabilistic programs**
    - Planning as inference
    - Sequential planning via recursion into the future
    - Multi-agent planning via recursion into other agents' minds
- **Why are PPLs uniquely suited for modeling agents?**
    - Agents are structured programs
    - Planning via nested conditional distributions
- **How do RL and PP relate?**
    - Algorithms vs models
    - Policy gradient as a PP

Where to go from here:
- [WebPPL](http://webppl.org) (webppl.org)
- [AgentModels](http://agentmodels.org) (agentmodels.org)
- andreas@ought.com
