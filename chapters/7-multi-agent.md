---
layout: chapter
title: Multi-agent models
description: Connection between recursively simulating self (sequential decisions) and recursive simulation of other agents (strategic reasoning, coordination). Schelling games, tic-tac-toe, a simple natural-language example, and inducation puzzles.
is_section: true
---

In this chapter, we will look at models that involve multiple agents reasoning about each other.
This chapter is based on reft:stuhlmueller2013reasoning.

## Schelling coordination games

We start with a simple Schelling coordination game: Alice and Bob are trying to meet up at one of two bars and have to decide between the popular one and the unpopular one.

Let's first consider only Alice:

~~~~
var locationPrior = function() {
  if (flip(.55)) {
    return 'popular-bar';
  } else {
    return 'unpopular-bar';
  }
};

var alice = function() {
  return Enumerate(function(){
    var myLocation = locationPrior();
    return myLocation;
  })
};

viz.auto(alice());
~~~~

Now Alice thinking about Bob:

~~~~
var locationPrior = function() {
  if (flip(.55)) {
    return 'popular-bar';
  } else {
    return 'unpopular-bar';
  }
};

var alice = function() {
  return Enumerate(function(){
    var myLocation = locationPrior();
    var bobLocation = sample(bob());
    condition(myLocation === bobLocation);
    return myLocation;
  });
};

var bob = function() {
  return Enumerate(function(){
    var myLocation = locationPrior();
    return myLocation;
  });
};

viz.auto(alice());
~~~~

Now Bob and Alice recursively, also adding caching and a depth parameter (to avoid infinite recursion):

~~~~
var locationPrior = function() {
  if (flip(.55)) {
    return 'popular-bar';
  } else {
    return 'unpopular-bar';
  }
}

var alice = dp.cache(function(depth) {
  return Enumerate(function(){
    var myLocation = locationPrior();
    var bobLocation = sample(bob(depth - 1));
    condition(myLocation === bobLocation);
    return myLocation;
  });
});

var bob = dp.cache(function(depth) {
  return Enumerate(function(){
    var myLocation = locationPrior();
    if (depth === 0) {
      return myLocation;
    } else {
      var aliceLocation = sample(alice(depth));
      condition(myLocation === aliceLocation);
      return myLocation;
    }
  });
});

viz.auto(alice(10))
~~~~

## Language understanding

Literal interpretation:

~~~~
var statePrior = function() {
  return uniformDraw([0, 1, 2, 3]);
};

var literalMeanings = {
  allSprouted: function(state) { return state === 3; },
  someSprouted: function(state) { return state > 0; },
  noneSprouted: function(state) { return state === 0; }
};

var sentencePrior = function() {
  return uniformDraw(['allSprouted', 'someSprouted', 'noneSprouted']);
};

var literalListener = function(sentence) {
  return Enumerate(function(){
    var state = statePrior();
    var meaning = literalMeanings[sentence];
    condition(meaning(state));
    return state;
  })
};

viz.auto(literalListener('someSprouted'));
~~~~

A pragmatic speaker, thinking about the literal listener:

~~~~
var alpha = 2;

var statePrior = function() {
  return uniformDraw([0, 1, 2, 3]);
};

var literalMeanings = {
  allSprouted: function(state) { return state === 3; },
  someSprouted: function(state) { return state > 0; },
  noneSprouted: function(state) { return state === 0; }
};

var sentencePrior = function() {
  return uniformDraw(['allSprouted', 'someSprouted', 'noneSprouted']);
};

var literalListener = function(sentence) {
  return Enumerate(function(){
    var state = statePrior();
    var meaning = literalMeanings[sentence];
    condition(meaning(state));
    return state;
  })
};

var speaker = function(state) {
  return Enumerate(function(){
    var sentence = sentencePrior();
    factor(alpha * literalListener(sentence).score([], state));
    return sentence;
  });
}

viz.auto(speaker(3));
~~~~

Pragmatic listener, thinking about speaker:

~~~~
var alpha = 2;

var statePrior = function() {
  return uniformDraw([0, 1, 2, 3]);
};

var literalMeanings = {
  allSprouted: function(state) { return state === 3; },
  someSprouted: function(state) { return state > 0; },
  noneSprouted: function(state) { return state === 0; }
};

var sentencePrior = function() {
  return uniformDraw(['allSprouted', 'someSprouted', 'noneSprouted']);
};

var literalListener = dp.cache(function(sentence) {
  return Enumerate(function(){
    var state = statePrior();
    var meaning = literalMeanings[sentence];
    condition(meaning(state));
    return state;
  })
});

var speaker = dp.cache(function(state) {
  return Enumerate(function(){
    var sentence = sentencePrior();
    factor(alpha * literalListener(sentence).score([], state));
    return sentence;
  });
});

var listener = dp.cache(function(sentence) {
  return Enumerate(function(){
    var state = statePrior();
    factor(speaker(state).score([], sentence));
    return state;
  })
});

viz.auto(listener('someSprouted'));
~~~~

## Game playing

We'll look at the two-player game tic-tac-toe.

Let's start with a prior on moves:

~~~~
var isValidMove = function(state, move) {
  return state[move.x][move.y] === '?';
};

var movePrior = dp.cache(function(state){
  return Enumerate(function(){
    var move = {
      x: randomInteger(3),
      y: randomInteger(3)
    };
    condition(isValidMove(state, move));
    return move;
  });
});

var startState = [
  ['?', 'o', '?'],
  ['?', 'x', 'x'],
  ['?', '?', '?']
];

viz.auto(movePrior(startState));
~~~~

Now let's add a deterministic transition function:

~~~~
var isValidMove = function(state, move) {
  return state[move.x][move.y] === '?';
};

var movePrior = dp.cache(function(state){
  return Enumerate(function(){
    var move = {
      x: randomInteger(3),
      y: randomInteger(3)
    };
    condition(isValidMove(state, move));
    return move;
  });
});

var assign = function(obj, k, v) {
  var newObj = _.clone(obj);
  return Object.defineProperty(newObj, k, {value: v})
};

var transition = function(state, move, player) {
  var newRow = assign(state[move.x], move.y, player);
  return assign(state, move.x, newRow);
};

var startState = [
  ['?', 'o', '?'],
  ['?', 'x', 'x'],
  ['?', '?', '?']
];

transition(startState, {x: 1, y: 0}, 'o')
~~~~

The next step would be to check win conditions...

~~~~
var act = dp.cache(function(state, player) {
  return Enumerate(function(){
    var action = actionPrior();
    var outcome = simulate(state, action, player);
    factor(utility(outcome, player));
    return action;
  });
});

var simulate = dp.cache(function(state, action, player) {
  var nextState = transition(state, action, player);
  if (isTerminal(nextState)) {
    return nextState;
  } else {
    var nextPlayer = otherPlayer(player);
    var nextAction = act(nextState, nextPlayer);
    return simulate(nextState, nextAction, nextPlayer);
  }
});

var startState = [
  ['?', 'o', '?'],
  ['?', 'x', 'x'],
  ['?', 'o', '?']
];

act(startState, 'x');
~~~~

## Induction puzzles

Blue-eyed islanders:

~~~~
var alpha = 2;

var agent = dp.cache(function(state) {
  return Enumerate(function(){
    var myBlueEyes = flip(baserate) ? 1 : 0;
    var totalBlueEyes = myBlueEyes + state.othersBlueEyes;
    condition(totalBlueEyes > 0);
    var raisedHandDist = simulate(/* */);
    factor(alpha * raisedHandDist.score(state.raisedHands));
    return myBlueEyes;
  });
});

var getRaisedHands(state) {
  // ...
};

var simulate = dp.cache(function(state) {
  if (state.start >= state.end) {
    return state.raisedHands;
  } else {
    var nextState = {
      start: state.start + 1,
      end: state.end,
      raisedHands: getRaisedHands(state),
      trueBlueEyes: state.trueBlueEyes
    };
    return simulate(nextState);
  }
});
~~~~