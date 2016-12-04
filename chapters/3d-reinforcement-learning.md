---
layout: chapter
title: "Reinforcement Learning to Learn MDPs"
description: RL vs. optimal Bayesian approach to Bandits, Softmax Greedy, Posterior Sampling for Bandits and MDPs, Q-learning and Policy Gradient
---

## Introduction

The previous chapter introduced POMDPs: decision problems where some features of the environment are initially unknown to the agent but can be learned by observation. We showed how to compute optimal Bayesian behavior for POMDPs. Unfortunately, this computation is infeasible for all but the simplest POMDPs. In practice, many POMDP problems can be solved heuristically using "Reinforcement Learning" (RL). RL algorithms are conceptually simple, scalable and effective both in discrete and continuous state spaces. They are central to achieving state-of-the-art performance in sequential decision problems in AI, e.g. in playing Go [cite], in playing videogames from raw pixels [cite], and in robotic control [cite]. 

## Reinforcement Learning for Bandits
The previous chapter showed how the optimal POMDP agent solves Bandit problems. Here we apply Reinforcement Learning to Bandits.

### Softmax Greedy Agent
We start with a "greedy" agent with softmax noise, which is similar to the well-known "Epsilon Greedy" agent for Bandits. The Softmax Greedy agent updates beliefs about the hidden state in the same way as the POMDP agent. Yet instead of making sequential plans that balance exploration (e.g. making informative observations) with exploitation (gaining high reward), the Greedy agent takes the action with highest *immediate* expected return[^greedy]. Here we implement the Greedy agent on Bernoulli Bandits. We measure the agent's performance by computing the *cumulative regret* over time. The regret for an action is the difference in expected returns between the action and the objective best action[^regret].

[^greedy]:In a later chapter, we implement a more general Greedy/Myopic agent by extending the POMDP agent. Here we implement the Greedy agent from scratch and apply it to Bernoulli Bandits.

[^regret]:The regret is a standard frequentist metric for performance. Bayesian metrics, which take into account the agent's priors, can also be defined but are beyond the scope of this chapter. 


~~~~
///fold:
var cumsum = function (xs) {
  var acf = function (n, acc) { return acc.concat( (acc.length > 0 ? acc[acc.length-1] : 0) + n); }
  return reduce(acf, [], xs.reverse());
  }
  ///
  

// Define Bandit problem

// Pull arm0 or arm1
var actions = [0, 1];

// Given a state (a coin-weight p for each arm), sample reward
var observeStateAction = function(state, action){
  var armToCoinWeight = state;
  return sample(Bernoulli({p : armToCoinWeight[action]})) 
};


// Greedy agent for Bandits
var makeGreedyBanditAgent = function(params) {
  var priorBelief = params.priorBelief;

  // Update belief about coin-weights observed reward
  var updateBelief = function(belief, observation, action){
    return Infer({ model() {
      var armToCoinWeight = sample(belief);
      condition( observation === observeStateAction(armToCoinWeight, action))
      return armToCoinWeight;
    }});
  };
  
  // Evaluate arms by expected coin-weight
  var expectedReward = function(belief, action){
    return expectation(Infer( { model() {
      var armToCoinWeight = sample(belief);
      return armToCoinWeight[action];
    }}))
  }

  // Choose by softmax over expected reward
  var act = dp.cache(
    function(belief) {
      return Infer({ model() {
        var action = uniformDraw(actions);
        factor(params.alpha * expectedReward(belief, action))
        return action;
      }});
    });

  return { params, act, updateBelief };
};

// Run Bandit problem
var simulate = function(armToCoinWeight, totalTime, agent) {
  var act = agent.act;
  var updateBelief = agent.updateBelief;
  var priorBelief = agent.params.priorBelief;

  var sampleSequence = function(timeLeft, priorBelief, action) {
    var observation = observeStateAction(armToCoinWeight, action);
    var belief = ((action === 'noAction') ? priorBelief :
                  updateBelief(priorBelief, observation, action))
    var action = sample(act(belief));

    return (timeLeft === 0) ? [action] : 
    [action].concat(sampleSequence(timeLeft-1, belief, action));
  };
  return sampleSequence(totalTime, priorBelief, 'noAction');
};


// Agent params
var alpha = 30
var priorBelief = Infer({  model () {
  var p0 = uniformDraw([.1, .3, .5, .6, .7, .9]);
  var p1 = uniformDraw([.1, .3, .5, .6, .7, .9]);
  return { 0:p0, 1:p1};
} });

// Bandit params
var numberTrials = 500;
var armToCoinWeight = { 0: 0.5, 1: 0.6 };

var agent = makeGreedyBanditAgent({alpha, priorBelief});
var trajectory = simulate(armToCoinWeight, numberTrials, agent);
var randomTrajectory = repeat(numberTrials, function(){return uniformDraw([0,1]);})

// Agent performance
var regret = function(arm) { 
  var bestCoinWeight = _.max(_.values(armToCoinWeight))
  return bestCoinWeight - armToCoinWeight[arm];
};
          

var trialToRegret = map(regret,trajectory);
var trialToRegretRandom = map(regret, randomTrajectory)
var ys = cumsum(map(regret, trajectory))

print('Number of trials: ' + numberTrials);
print('Total regret: [Greedy, Random]', [sum(trialToRegret), sum(trialToRegretRandom)])
print('Arms pulled: ' +  trajectory);

viz.line(_.range(ys.length), ys, {xLabel:'Time', yLabel:'Cumulative regret'});
~~~~


>*Exercise*:

> 1. Set the softmax noise to be low. How well does the Greedy Softmax agent do? Explain why. Keeping the noise low, modify the agent's priors to be overly "optimistic" about the expected reward of each arm (without changing the support of the prior distribution). How does this optimism change the agent's performance? Explain why. (This idea is known as "optimism in the face of uncertainty" in the RL literature.)

> 2. Modify the agent so that the softmax noise is low and the agent has a "bad" prior (i.e. one that assigns a low probability to the truth) that is not optimistic. Will the agent eventually learn the optimal policy? How many trials does it take on average?


Thompson sampling:

~~~~
///fold: Bandit problem is defined as above

// Pull arm0 or arm1
var actions = [0, 1, 2];

// Use latent "armToPrize" mapping in state to
// determine which prize agent gets
var transition = function(state, action){
  var newTimeLeft = state.timeLeft - 1;
  var armER = state.armToExpectedReward[action];
  return extend(state, {
    score : sample(Bernoulli({p : armER })), 
    timeLeft: newTimeLeft,
    terminateAfterAction: newTimeLeft == 1
  });
};

// After pulling an arm, agent observes associated prize
var observe = function(state){
  return state.score;
};

// Defining the POMDP agent

// Agent params include utility function and initial belief (*priorBelief*)

var makeAgent = function(params) {
  var utility = params.utility;

  // Implements *Belief-update formula* in text
  var updateBelief = function(belief, observation, action){
    return Infer({ model() {
      var state = sample(belief);
      var predictedNextState = transition(state, action);
      var predictedObservation = observe(predictedNextState);
      condition(_.isEqual(predictedObservation, observation));
      return predictedNextState;
    }});
  };

  var act = dp.cache(
    function(belief) {
      var thompsonState = sample(belief);
      return Infer({ model() {
        var action = uniformDraw(actions);

        factor(1000 * utility(thompsonState, action));
        return action;
      }});
    });

  return { params, act, updateBelief };
};

var cumsum = function (xs) {
  var acf = function (n, acc) { return acc.concat( (acc.length > 0 ? acc[acc.length-1] : 0) + n); }
  return reduce(acf, [], xs.reverse());
}

var simulate = function(startState, agent) {
  var act = agent.act;
  var updateBelief = agent.updateBelief;
  var priorBelief = agent.params.priorBelief;

  var sampleSequence = function(state, priorBelief, action) {
    var observation = observe(state);
    var belief = ((action === 'noAction') ? priorBelief : 
                  updateBelief(priorBelief, observation, action));
    var action = sample(act(belief));
    var output = [[state, action]];

    if (state.terminateAfterAction){
      return output;
    } else {
      var nextState = transition(state, action);
      return output.concat(sampleSequence(nextState, belief, action));
    }
 
  };
  // Start with agent's prior and a special "null" action
  return sampleSequence(startState, priorBelief, 'noAction');
};



//-----------
// Construct the agent

var utility = function(state, action) {
  return state.armToExpectedReward[action];
};


// Define true startState (including true *armToPrize*) and
// alternate possibility for startState (see Figure 2)

var numberTrials = 100;
var startState = { 
  score: 0,
  timeLeft: numberTrials + 1, 
  terminateAfterAction: false,
  armToExpectedReward: [0.3, .5, 0.9]
};

var regret = function (a) { return .9 - utility(startState, a); }

// Agent's prior
var priorBelief = Infer({  model () {
  var p0 = uniformDraw([.1, .3, .5, .7, .9]);
  var p1 = uniformDraw([.1, .3, .5, .7, .9]);
  var p2 = uniformDraw([.1, .3, .5, .7, .9]);

  return extend(startState, { armToExpectedReward :  [p0, p1, p2] });
} });


var params = { utility: utility, priorBelief: priorBelief };
var agent = makeAgent(params);
var trajectory = simulate(startState, agent);

print('Number of trials: ' + numberTrials);
print('Arms pulled: ' +  map(second, trajectory));

var ys = cumsum(map(regret, map(second, trajectory)))
viz.line(_.range(ys.length), ys);
~~~~


### Footnotes

