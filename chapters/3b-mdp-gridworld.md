---
layout: chapter
title: "MDPs and Gridworld in WebPPL"
description: We extend the previous setup to noisy actions (softmax) and transitions, and introduce policies and expected action values.
---

This chapter explores some key features of MDPs: stochastic dynamics, stochastic policies, and value functions.

### Hiking in Gridworld

We begin by introducing a new gridworld MDP:

> **Hiking Problem**:
>Suppose that Alice is hiking. There are two peaks nearby, denoted "West" and "East". The peaks provide different views and Alice must choose between them. South of Alice's starting position is a steep hill. Falling down the hill would result in painful (but non-fatal) injury and end the hike early.

We represent Alice's hiking problem with a Gridworld similar to Bob's Restaurant Choice example. The peaks are terminal states, providing different utilities. The steep hill is represented by a row of terminal state, each with identical negative utility. Each timestep before Alice reaches a terminal state incurs a "time cost", which is negative to represent the fact that Alice prefers a shorter hike. <!-- TODO might be good to indicate on plot that the steep hills are bad -->

<!-- draw_hike -->
~~~~
var H = { name: 'Hill' };
var W = { name: 'West' };
var E = { name: 'East' };
var ___ = ' ';

var grid = [
  [___, ___, ___, ___, ___],
  [___, '#', ___, ___, ___],
  [___, '#',  W , '#',  E ],
  [___, ___, ___, ___, ___],
  [ H ,  H ,  H ,  H ,  H ]
];

var start = [0, 1];

var mdp = makeGridWorldMDP({ grid, start });

viz.gridworld(mdp.world, { trajectory: [mdp.startState] });
~~~~

We start with a *deterministic* transition function. In this case, Alice's risk of falling down the steep hill is solely due to softmax noise in her action choice (which is minimal in this case). The agent model is the same as the one at the end of [Chapter III.1](/chapters/3a-mdp.html). We place the functions `act`, `expectedUtility` in a function `makeMDPAgent`. The following codebox defines this function and we use it later on without defining it (since it's in the `webppl-agents` library).

<!-- define_agent_simulate -->
~~~~
// Set up agent structure

var makeMDPAgent = function(params, world) {
  var stateToActions = world.stateToActions;
  var transition = world.transition;
  var utility = params.utility;
  var alpha = params.alpha;

  var act = dp.cache(
    function(state) {
      return Infer({ model() {
        var action = uniformDraw(stateToActions(state));
        var eu = expectedUtility(state, action);
        factor(alpha * eu);
        return action;
      }});
    });

  var expectedUtility = dp.cache(
    function(state, action){
      var u = utility(state, action);
      if (state.terminateAfterAction){
        return u;
      } else {
        return u + expectation(Infer({ model() {
          var nextState = transition(state, action);
          var nextAction = sample(act(nextState));
          return expectedUtility(nextState, nextAction);
        }}));
      }
    });

  return { params, expectedUtility, act };
};

var simulate = function(startState, world, agent) {
  var act = agent.act;
  var transition = world.transition;
  var sampleSequence = function(state) {
    var action = sample(act(state));
    var nextState = transition(state, action);
    if (state.terminateAfterAction) {
      return [state];
    } else {
      return [state].concat(sampleSequence(nextState));
    }
  };
  return sampleSequence(startState);
};


// Set up world

var makeHikeMDP = function(options) {
  var H = { name: 'Hill' };
  var W = { name: 'West' };
  var E = { name: 'East' };
  var ___ = ' ';
  var grid = [
    [___, ___, ___, ___, ___],
    [___, '#', ___, ___, ___],
    [___, '#',  W , '#',  E ],
    [___, ___, ___, ___, ___],
    [ H ,  H ,  H ,  H ,  H ]
  ];
  return makeGridWorldMDP(_.assign({ grid }, options));
};

var mdp = makeHikeMDP({
  start: [0, 1],
  totalTime: 12,
  transitionNoiseProbability: 0
});

var makeUtilityFunction = mdp.makeUtilityFunction;


// Create parameterized agent

var utility = makeUtilityFunction({
  East: 10,
  West: 1,
  Hill: -10,
  timeCost: -.1
});
var agent = makeMDPAgent({ utility, alpha: 1000 }, mdp.world);


// Run agent on world

var trajectory = simulate(mdp.startState, mdp.world, agent);


viz.gridworld(mdp.world, { trajectory });
~~~~

>**Exercise**: Adjust the parameters of `utilityTable` in order to produce the following behaviors:

>1. The agent goes directly to "West".
>2. The agent takes the long way around to "West".
>3. The agent sometimes goes to the Hill at $$[1,0]$$. Try to make this outcome as likely as possible.
<!-- 3 is obtained by making timeCost positive and Hill better than alternatives -->


### Hiking with stochastic transitions

Imagine that the weather is very wet and windy. As a result, Alice will sometimes intend to go one way but actually go another way (because she slips in the mud). In this case, the shorter route to the peaks might be too risky for Alice.

To model bad weather, we assume that at every timestep, there is a constant independent probability `transitionNoiseProbability` of the agent moving orthogonally to their intended direction. The independence assumption is unrealistic (if a location is slippery at one timestep it is more likely slippery the next), but it is simple and satisfies the Markov assumption for MDPs.

Setting `transitionNoiseProbability=0.1`, the agent's first action is now to move "up" instead of "right".

~~~~
///fold: makeHikeMDP
var makeHikeMDP = function(options) {
  var H = { name: 'Hill' };
  var W = { name: 'West' };
  var E = { name: 'East' };
  var ___ = ' ';
  var grid = [
    [___, ___, ___, ___, ___],
    [___, '#', ___, ___, ___],
    [___, '#',  W , '#',  E ],
    [___, ___, ___, ___, ___],
    [ H ,  H ,  H ,  H ,  H ]
  ];
  return makeGridWorldMDP(_.assign({ grid }, options));
};
///

// Set up world

var mdp = makeHikeMDP({
  start: [0, 1],
  totalTime: 13,
  transitionNoiseProbability: 0.1  // <- NEW
});


// Create parameterized agent

var makeUtilityFunction = mdp.makeUtilityFunction;
var utility = makeUtilityFunction({
  East: 10,
  West: 1,
  Hill: -10,
  timeCost: -.1
});
var agent = makeMDPAgent({ utility, alpha: 100 }, mdp.world);


// Generate a single trajectory, draw

var trajectory = simulateMDP(mdp.startState, mdp.world, agent, 'states');
viz.gridworld(mdp.world, { trajectory });


// Generate 100 trajectories, plot distribution on lengths

var trajectoryDist = Infer({
  model() {
    var trajectory = simulateMDP(mdp.startState, mdp.world, agent);
    return { trajectoryLength: trajectory.length }
  },
  method: 'forward',
  samples: 100
});

viz(trajectoryDist);
~~~~

>**Exercise:**

>1. Keeping `transitionNoiseProbability=0.1`, find settings for `utilityTable` such that the agent goes "right" instead of "up".
>2. Set `transitionNoiseProbability=0.01`. Change a single parameter in `utilityTable` such that the agent goes "right" (there are multiple ways to do this).
<!-- put up timeCost to -1 or so -->

### Noisy transitions vs. Noisy agents

It's important to distinguish noise in the transition function from the softmax noise in the agent's selection of actions. Noise (or "stochasticity") in the transition function is a representation of randomness in the world. This is easiest to think about in games of chance[^noise]. In a game of chance (e.g. slot machines or poker) rational agents will take into account the randomness in the game. By contrast, softmax noise is a property of an agent. For example, we can vary the behavior of otherwise identical agents by varying their parameter $$\alpha$$.

Unlike transition noise, softmax noise has little influence on the agent's planning for the Hiking Problem. Since it's so bad to fall down the hill, the softmax agent will rarely do so even if they take the short route. The softmax agent is like a person who takes inefficient routes when stakes are low but "pulls themself together" when stakes are high.

[^noise]: An agent's world model might treat a complex set of deterministic rules as random. In this sense, agents will vary in whether they represent an MDP as stochastic or not. We won't consider that case in this tutorial.

>**Exercise:** Use the codebox below to explore different levels of softmax noise. Find a setting of `utilityTable` and `alpha` such that the agent goes to West and East equally often and nearly always takes the most direct route to both East and West. Included below is code for simulating many trajectories and returning the trajectory length. You can extend this code to measure whether the route taken by the agent is direct or not. (Note that while the softmax agent here is able to "backtrack" or return to its previous location, in later Gridworld examples we disalllow backtracking as a possible action).

~~~~
///fold: makeHikeMDP, set up world
var makeHikeMDP = function(options) {
  var H = { name: 'Hill' };
  var W = { name: 'West' };
  var E = { name: 'East' };
  var ___ = ' ';
  var grid = [
    [___, ___, ___, ___, ___],
    [___, '#', ___, ___, ___],
    [___, '#',  W , '#',  E ],
    [___, ___, ___, ___, ___],
    [ H ,  H ,  H ,  H ,  H ]
  ];
  return makeGridWorldMDP(_.assign({ grid }, options));
};

var mdp = makeHikeMDP({
  start: [0, 1],
  totalTime: 13,
  transitionNoiseProbability: 0.1
});

var world = mdp.world;
var startState = mdp.startState;
var makeUtilityFunction = mdp.makeUtilityFunction;
///

// Create parameterized agent
var utility = makeUtilityFunction({
  East: 10,
  West: 1,
  Hill: -10,
  timeCost: -.1
});
var alpha = 1;  // <- SOFTMAX NOISE
var agent = makeMDPAgent({ utility, alpha }, world);

// Generate a single trajectory, draw
var trajectory = simulateMDP(startState, world, agent, 'states');
viz.gridworld(world, { trajectory });

// Generate 100 trajectories, plot distribution on lengths
var trajectoryDist = Infer({
  model() {
    var trajectory = simulateMDP(startState, world, agent);
    return { trajectoryLength: trajectory.length }
  },
  method: 'forward',
  samples: 100
});
viz(trajectoryDist);
~~~~


### Stochastic transitions: plans and policies

We return to the case of a stochastic environment with very low softmax action noise. In a stochastic environment, the agent sometimes finds themself in a state they did not intend to reach. The functions `agent` and `expectedUtility` (inside `makeMDPAgent`) implicitly compute the expected utility of actions for every possible future state, including states that the agent will try to avoid. In the MDP literature, this function from states and remaining time to actions is called a *policy*. (For infinite-horizon MDPs, policies are functions from states to actions.) Since policies take into account every possible contingency, they are quite different from the everyday notion of a plan.

Consider the example from above where the agent takes the long route because of the risk of falling down the hill. If we generate a single trajectory for the agent, they will likely take the long route. However, if we generated many trajectories, we would sometimes see the agent move "right" instead of "up" on their first move. Before taking this first action, the agent implicitly computes what they *would* do if they end up moving right. To find out what they would do, we can artificially start the agent in $[1,1]$ instead of $[0,1]$:

<!-- policy -->
~~~~
///fold: makeHikeMDP
var makeHikeMDP = function(options) {
  var H = { name: 'Hill' };
  var W = { name: 'West' };
  var E = { name: 'East' };
  var ___ = ' ';
  var grid = [
    [___, ___, ___, ___, ___],
    [___, '#', ___, ___, ___],
    [___, '#',  W , '#',  E ],
    [___, ___, ___, ___, ___],
    [ H ,  H ,  H ,  H ,  H ]
  ];
  return makeGridWorldMDP(_.assign({ grid }, options));
};
///

// Parameters for world
var mdp = makeHikeMDP({
  start: [1, 1],  // Previously: [0, 1]
  totalTime: 11,             // Previously: 12
  transitionNoiseProbability: 0.1
});
var makeUtilityFunction = mdp.makeUtilityFunction;

// Parameters for agent
var utility = makeUtilityFunction({ 
  East: 10, 
  West: 1,
  Hill: -10,
  timeCost: -.1
});
var agent = makeMDPAgent({ utility, alpha: 1000 }, mdp.world);
var trajectory = simulateMDP(mdp.startState, mdp.world, agent, 'states');

viz.gridworld(mdp.world, { trajectory });
~~~~

Extending this idea, we can display the expected values of each action the agent *could have taken* during their trajectory. These expected values numbers are analogous to state-action Q-values in infinite-horizon MDPs.

The expected values were already being computed implicitly; we now use `getExpectedUtilitiesMDP` to access them. The displayed numbers in each grid cell are the expected utilities of moving in the corresponding directions. For example, we can read off how close the agent was to taking the short route as opposed to the long route. (Note that if the difference in expected utility between two actions is small then a noisy agent will take each of them with nearly equal probability).

~~~~
///fold: makeBigHikeMDP, getExpectedUtilitiesMDP
var makeBigHikeMDP = function(options) {
  var H = { name: 'Hill' };
  var W = { name: 'West' };
  var E = { name: 'East' };
  var ___ = ' ';
  var grid = [
    [___, ___, ___, ___, ___, ___],
    [___, ___, ___, ___, ___, ___],
    [___, ___, '#', ___, ___, ___],
    [___, ___, '#',  W , '#',  E ],
    [___, ___, ___, ___, ___, ___],
    [ H ,  H ,  H ,  H ,  H ,  H ]
  ];
  return makeGridWorldMDP(_.assign({ grid }, options));
};

// trajectory must consist only of states. This can be done by calling
// *simulate* with an additional final argument 'states'.
var getExpectedUtilitiesMDP = function(stateTrajectory, world, agent) {
  var eu = agent.expectedUtility;
  var actions = world.actions;
  var getAllExpectedUtilities = function(state) {
    var actionUtilities = map(
      function(action){ return eu(state, action); },
      actions);
    return [state, actionUtilities];
  };
  return map(getAllExpectedUtilities, stateTrajectory);
};
///

// Long route is better, agent takes long route

var mdp = makeBigHikeMDP({
  start: [1, 1],
  totalTime: 12,
  transitionNoiseProbability: 0.03
});
var makeUtilityFunction = mdp.makeUtilityFunction;

var utility = makeUtilityFunction({
  East: 10,
  West: 7,
  Hill : -40,
  timeCost: -0.4
});
var agent = makeMDPAgent({ utility, alpha: 100 }, mdp.world);

var trajectory = simulateMDP(mdp.startState, mdp.world, agent, 'states');
var actionExpectedUtilities = getExpectedUtilitiesMDP(trajectory, mdp.world, agent);

viz.gridworld(mdp.world, { trajectory, actionExpectedUtilities });
~~~~

So far, our agents all have complete knowledge about the state of the world. In the [next chapter](/chapters/3c-pomdp.html), we will explore partially observable worlds.

<br>

### Footnotes
