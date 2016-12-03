---
layout: chapter
title: Multi-agent models
description: Schelling coordination games, tic-tac-toe, and a simple natural-language example.
is_section: true
---

In this chapter, we will look at models that involve multiple agents reasoning about each other.
This chapter is based on reft:stuhlmueller2013reasoning.

## Schelling coordination games

We start with a simple [Schelling coordination game](http://lesswrong.com/lw/dc7/nash_equilibria_and_schelling_points/). Alice and Bob are trying to meet up but have lost their phones and have no way to contact each other. There are two local bars: the popular bar and the unpopular one.

Let's first consider how Alice would choose a bar (if she was not taking Bob into account):

~~~~
var locationPrior = function() {
  if (flip(.55)) {
    return 'popular-bar';
  } else {
    return 'unpopular-bar';
  }
};

var alice = function() {
  return Infer({ model() {
    var myLocation = locationPrior();
    return myLocation;
  }});
};

viz(alice());
~~~~

But Alice wants to be at the same bar as Bob. We extend our model of Alice to include this:

~~~~
var locationPrior = function() {
  if (flip(.55)) {
    return 'popular-bar';
  } else {
    return 'unpopular-bar';
  }
};

var alice = function() {
  return Infer({ model() {
    var myLocation = locationPrior();
    var bobLocation = sample(bob());
    condition(myLocation === bobLocation);
    return myLocation;
  }});
};

var bob = function() {
  return Infer({ model() {
    var myLocation = locationPrior();
    return myLocation;
  }});
};

viz(alice());
~~~~

Now Bob and Alice are thinking recursively about each other. We add caching (to avoid repeated computations) and a depth parameter (to avoid infinite recursion):

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

viz(alice(10));
~~~~

>**Exercise**: Change the example to the setting where Bob wants to avoid Alice instead of trying to meet up with her, and Alice knows this. How do the predictions change as the reasoning depth grows? How would you model the setting where Alice doesn't know that Bob wants to avoid her?

>**Exercise**: Would any of the answers to the previous exercise change if recursive reasoning could terminate not just at a fixed depth, but also at random?


## Game playing

We'll look at the two-player game tic-tac-toe [^tictactoeimg]:

>*Figure 1:* Tic-tac-toe. (Image source: [Wikipedia](https://en.wikipedia.org/wiki/Tic-tac-toe#/media/File:Tic-tac-toe-game-1.svg))

<img src="/assets/img/tic-tac-toe-game-1.svg"/>

Let's start with a prior on moves:

~~~~
var isValidMove = function(state, move) {
  return state[move.x][move.y] === '?';
};

var movePrior = dp.cache(function(state){
  return Infer({ model() {
    var move = {
      x: randomInteger(3),
      y: randomInteger(3)
    };
    condition(isValidMove(state, move));
    return move;
  }});
});

var startState = [
  ['?', 'o', '?'],
  ['?', 'x', 'x'],
  ['?', '?', '?']
];

viz.table(movePrior(startState));
~~~~

Now let's add a deterministic transition function:

~~~~
///fold: isValidMove, movePrior
var isValidMove = function(state, move) {
  return state[move.x][move.y] === '?';
};

var movePrior = dp.cache(function(state){
  return Infer({ model() {
    var move = {
      x: randomInteger(3),
      y: randomInteger(3)
    };
    condition(isValidMove(state, move));
    return move;
  }});
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

transition(startState, {x: 1, y: 0}, 'o');
~~~~

We need to be able to check if a player has won:

~~~~
///fold: movePrior, transition
var isValidMove = function(state, move) {
  return state[move.x][move.y] === '?';
};

var movePrior = dp.cache(function(state){
  return Infer({ model() {
    var move = {
      x: randomInteger(3),
      y: randomInteger(3)
    };
    condition(isValidMove(state, move));
    return move;
  }});
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

hasWon(startState, 'x');
~~~~

Now let's implement an agent that chooses a single action, but can't plan ahead:

~~~~
///fold: movePrior, transition, hasWon
var isValidMove = function(state, move) {
  return state[move.x][move.y] === '?';
};

var movePrior = dp.cache(function(state){
  return Infer({ model() {
    var move = {
      x: randomInteger(3),
      y: randomInteger(3)
    };
    condition(isValidMove(state, move));
    return move;
  }});
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
  return Infer({ model() {
    var move = sample(movePrior(state));
    var eu = expectation(Infer({ model() {
      var outcome = transition(state, move, player);
      return utility(outcome, player);
    }}));
    factor(eu);    
    return move;
  }});
});

var startState = [
  ['o', 'o', '?'],
  ['?', 'x', 'x'],
  ['?', '?', '?']
];

viz.table(act(startState, 'x'));
~~~~

And now let's include planning:

~~~~
///fold: movePrior, transition, hasWon, utility, isTerminal
var isValidMove = function(state, move) {
  return state[move.x][move.y] === '?';
};

var movePrior = dp.cache(function(state){
  return Infer({ model() {
    var move = {
      x: randomInteger(3),
      y: randomInteger(3)
    };
    condition(isValidMove(state, move));
    return move;
  }});
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

var isComplete = function(state) {
  return all(
    function(x){
      return x != '?';
    },
    _.flatten(state));
}

var isTerminal = function(state) {
  return hasWon(state, 'x') || hasWon(state, 'o') || isComplete(state);  
};
///

var otherPlayer = function(player) {
  return (player === 'x') ? 'o' : 'x';
};

var act = dp.cache(function(state, player) {
  return Infer({ model() {
    var move = sample(movePrior(state));
    var eu = expectation(Infer({ model() {
      var outcome = simulate(state, move, player);
      return utility(outcome, player);
    }}));
    factor(eu);    
    return move;
  }});
});

var simulate = function(state, action, player) {
  var nextState = transition(state, action, player);
  if (isTerminal(nextState)) {
    return nextState;
  } else {
    var nextPlayer = otherPlayer(player);
    var nextAction = sample(act(nextState, nextPlayer));
    return simulate(nextState, nextAction, nextPlayer);
  }
};

var startState = [
  ['o', '?', '?'],
  ['?', '?', 'x'],
  ['?', '?', '?']
];

var actDist = act(startState, 'o');

viz.table(actDist);
~~~~

## Language understanding

A model of pragmatic language interpretation: The speaker chooses a sentence conditioned on the listener inferring the intended state of the world when hearing this sentence; the listener chooses an interpretation conditioned on the speaker selecting the given utterance when intending this meaning.

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
  return Infer({ model() {
    var state = statePrior();
    var meaning = literalMeanings[sentence];
    condition(meaning(state));
    return state;
  }});
};

viz(literalListener('someSprouted'));
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
  return Infer({ model() {
    var state = statePrior();
    var meaning = literalMeanings[sentence];
    condition(meaning(state));
    return state;
  }});
};

var speaker = function(state) {
  return Infer({ model() {
    var sentence = sentencePrior();
    factor(alpha * literalListener(sentence).score(state));
    return sentence;
  }});
}

viz(speaker(3));
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
  return Infer({ model() {
    var state = statePrior();
    var meaning = literalMeanings[sentence];
    condition(meaning(state));
    return state;
  }});
});

var speaker = dp.cache(function(state) {
  return Infer({ model() {
    var sentence = sentencePrior();
    factor(alpha * literalListener(sentence).score(state));
    return sentence;
  }});
});

var listener = dp.cache(function(sentence) {
  return Infer({ model() {
    var state = statePrior();
    factor(speaker(state).score(sentence));
    return state;
  }});
});

viz(listener('someSprouted'));
~~~~

Next chapter: [How to use the WebPPL Agent Models library](/chapters/8-using-gridworld-library.html)

<br>

### Footnotes
