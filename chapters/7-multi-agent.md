---
layout: chapter
title: Multi-agent models
description: Connection between recursively simulating self (sequential decisions) and recursive simulation of other agents (strategic reasoning, coordination). Schelling games, tic-tac-toe, a simple natural-language example, and induction puzzles.
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
  return Infer({ method: 'enumerate' }, function(){
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
  return Infer({ method: 'enumerate' }, function(){
    var myLocation = locationPrior();
    var bobLocation = sample(bob());
    condition(myLocation === bobLocation);
    return myLocation;
  });
};

var bob = function() {
  return Infer({ method: 'enumerate' }, function(){
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
  return Infer({ method: 'enumerate' }, function(){
    var myLocation = locationPrior();
    var bobLocation = sample(bob(depth - 1));
    condition(myLocation === bobLocation);
    return myLocation;
  });
});

var bob = dp.cache(function(depth) {
  return Infer({ method: 'enumerate' }, function(){
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
  return Infer({ method: 'enumerate' }, function(){
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

///fold: statePrior, literalMeanings, sentencePrior
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
///

var literalListener = function(sentence) {
  return Infer({ method: 'enumerate' }, function(){
    var state = statePrior();
    var meaning = literalMeanings[sentence];
    condition(meaning(state));
    return state;
  })
};

var speaker = function(state) {
  return Infer({ method: 'enumerate' }, function(){
    var sentence = sentencePrior();
    factor(alpha * literalListener(sentence).score(state));
    return sentence;
  });
}

viz.auto(speaker(3));
~~~~

Pragmatic listener, thinking about speaker:

~~~~
var alpha = 2;

///fold: statePrior, literalMeanings, sentencePrior
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
///

var literalListener = dp.cache(function(sentence) {
  return Infer({ method: 'enumerate' }, function(){
    var state = statePrior();
    var meaning = literalMeanings[sentence];
    condition(meaning(state));
    return state;
  })
});

var speaker = dp.cache(function(state) {
  return Infer({ method: 'enumerate' }, function(){
    var sentence = sentencePrior();
    factor(alpha * literalListener(sentence).score(state));
    return sentence;
  });
});

var listener = dp.cache(function(sentence) {
  return Infer({ method: 'enumerate' }, function(){
    var state = statePrior();
    factor(speaker(state).score(sentence));
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
  return Infer({ method: 'enumerate' }, function(){
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
///fold: isValidMove, movePrior
var isValidMove = function(state, move) {
  return state[move.x][move.y] === '?';
};

var movePrior = dp.cache(function(state){
  return Infer({ method: 'enumerate' }, function(){
    var move = {
      x: randomInteger(3),
      y: randomInteger(3)
    };
    condition(isValidMove(state, move));
    return move;
  });
});
///

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

We need to be able to check if a player has won:

~~~~
///fold: movePrior, transition
var isValidMove = function(state, move) {
  return state[move.x][move.y] === '?';
};

var movePrior = dp.cache(function(state){
  return Infer({ method: 'enumerate' }, function(){
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
///

var diag1 = function(state) {
  return mapIndexed(function(i, x) {return x[i];}, state);
};

var diag2 = function(state) {
  var n = state.length;
  return mapIndexed(function(i, x) {return x[n - (i + 1)];}, state);
};

var hasWon = dp.cache(function(state, player) {
  var check = function(xs){
    return _.countBy(xs)[player] == xs.length;
  };
  return any(check, [
    state[0], state[1], state[2], // rows
    map(first, state), map(second, state), map(third, state), // cols
    diag1(state), diag2(state) // diagonals
  ]);
});

var startState = [
  ['?', 'o', '?'],
  ['x', 'x', 'x'],
  ['?', '?', '?']
];

hasWon(startState, 'x')
~~~~

Now let's implement an agent that chooses a single action, but can't plan ahead:

~~~~
///fold: movePrior, transition, hasWon
var isValidMove = function(state, move) {
  return state[move.x][move.y] === '?';
};

var movePrior = dp.cache(function(state){
  return Infer({ method: 'enumerate' }, function(){
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

var diag1 = function(state) {
  return mapIndexed(function(i, x) {return x[i];}, state);
};

var diag2 = function(state) {
  var n = state.length;
  return mapIndexed(function(i, x) {return x[n - (i + 1)];}, state);
};

var hasWon = dp.cache(function(state, player) {
  var check = function(xs){
    return _.countBy(xs)[player] == xs.length;
  };
  return any(check, [
    state[0], state[1], state[2], // rows
    map(first, state), map(second, state), map(third, state), // cols
    diag1(state), diag2(state) // diagonals
  ]);
});
///
var isDraw = function(state) {
  return !hasWon(state, 'x') && !hasWon(state, 'o');
};

var utility = function(state, player) {
  if (hasWon(state, player)) {
    return 10;
  } else if (isDraw(state)) {
    return 0;
  } else {
    return -10;
  }
};

var act = dp.cache(function(state, player) {
  return Infer({ method: 'enumerate' }, function(){
    var move = sample(movePrior(state));
    var outcome = transition(state, move, player);
    factor(utility(outcome, player));
    return move;
  });
});

var startState = [
  ['o', 'o', '?'],
  ['?', 'x', 'x'],
  ['?', '?', '?']
];

viz.auto(act(startState, 'o'))
~~~~

And now let's include planning:

~~~~
///fold: movePrior, transition, hasWon, utility, isTerminal
var isValidMove = function(state, move) {
  return state[move.x][move.y] === '?';
};

var movePrior = dp.cache(function(state){
  return Infer({ method: 'enumerate' }, function(){
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

var diag1 = function(state) {
  return mapIndexed(function(i, x) {return x[i];}, state);
};

var diag2 = function(state) {
  var n = state.length;
  return mapIndexed(function(i, x) {return x[n - (i + 1)];}, state);
};

var hasWon = dp.cache(function(state, player) {
  var check = function(xs){
    return _.countBy(xs)[player] == xs.length;
  };
  return any(check, [
    state[0], state[1], state[2], // rows
    map(first, state), map(second, state), map(third, state), // cols
    diag1(state), diag2(state) // diagonals
  ]);
});

var isDraw = function(state) {
  return !hasWon(state, 'x') && !hasWon(state, 'o');
};

var utility = function(state, player) {
  if (hasWon(state, player)) {
    return 10;
  } else if (isDraw(state)) {
    return 0;
  } else {
    return -10;
  }
};

var isTerminal = function(state) {
  return all(
    function(x){
      return x != '?';
    },
    _.flatten(state));
};
///

var otherPlayer = function(player) {
  return (player === 'x') ? 'o' : 'x';
};

var act = dp.cache(function(state, player) {
  return Infer({ method: 'enumerate' }, function(){
    var move = sample(movePrior(state));
    var outcome = simulate(state, move, player);
    factor(utility(outcome, player));
    return move;
  });
});

var simulate = dp.cache(function(state, action, player) {
  var nextState = transition(state, action, player);
  if (isTerminal(nextState)) {
    return nextState;
  } else {
    var nextPlayer = otherPlayer(player);
    var nextAction = sample(act(nextState, nextPlayer));
    return simulate(nextState, nextAction, nextPlayer);
  }
});

var startState = [
  ['?', '?', '?'],
  ['?', 'x', 'x'],
  ['o', '?', '?']
];

viz.auto(act(startState, 'o'))
~~~~

## Induction puzzles

Blue-eyed islanders:

~~~~
// FIXME:
// AssertionError: Expected marginal to be normalized, got: 
// {"2":{"val":2,"prob":null},"3":{"val":3,"prob":null},"4":{"val":4,"prob":null}}

var alpha = 2;

var assign = function(obj, k, v) {
  var newObj = _.clone(obj);
  return Object.defineProperty(newObj, k, {value: v})
};


var numAgents = 2;
var baserate = .045;

var agent = dp.cache(function(t, raisedHands, othersBlueEyes) {
  if (1 + othersBlueEyes < raisedHands) {
    return Infer({ method: 'enumerate' }, function(){
      return 1;
    })
  } else {
    return Infer({ method: 'enumerate' }, function(){
      var myBlueEyes = flip(baserate) ? 1 : 0;
      var totalBlueEyes = myBlueEyes + othersBlueEyes;
      condition(totalBlueEyes >= 0);
      condition(totalBlueEyes <= numAgents);      
      var outcome = runGame(0, t, 0, totalBlueEyes);
      condition(outcome == raisedHands);
      return myBlueEyes;
    });
  }
});

var getRaisedHands = function(t, raisedHands, trueBlueEyes) {
  var p1 = Math.exp(agent(t, raisedHands, trueBlueEyes - 1).score(1));
  var p2 = Math.exp(agent(t, raisedHands, trueBlueEyes).score(1));
  return binomial(p1, trueBlueEyes) + binomial(p2, numAgents - trueBlueEyes);
};

var runGame = function(start, end, raisedHands, trueBlueEyes) {
  if (start >= end) {
    return raisedHands;
  } else {
    var raisedHands = getRaisedHands(start, raisedHands, trueBlueEyes)
    return runGame(start + 1, end, raisedHands, trueBlueEyes);    
  }
};

viz.auto(Infer({ method: 'enumerate' }, function(){return runGame( 0, 2, 0, 2);}));
~~~~
