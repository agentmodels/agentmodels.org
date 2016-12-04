---
layout: chapter
title: "POMDPs and Reinforcement Learning"
description: 
---


 
## Introduction: Reinforcement Learning as specialized POMDPs

Softmax Greedy. TODO: simplify the code.

~~~~
///fold: Bandit problem is defined as above
var getArmExpectedReward = function(belief, action){
  return expectation(Infer( { model() {
    var belief = sample(belief);
    return belief.armToExpectedReward[action];
  }}))
}


// Pull arm0 or arm1
var actions = [0, 1];

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
      return Infer({ model() {
        var action = uniformDraw(actions);
        factor(100 * getArmExpectedReward(belief, action))
        return action;
      }});
    });

  return { params, act, updateBelief };
};

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

var numberTrials = 1000;
var startState = { 
  score: 0,
  timeLeft: numberTrials + 1, 
  terminateAfterAction: false,
  armToExpectedReward: { 0: 0.5, 1: 0.55 }
};

// Agent's prior
var priorBelief = Infer({  model () {
  var p0 = uniformDraw([.1, .3, .5, .55, .7, .9]);
  var p1 = uniformDraw([.1, .3, .5, .55, .7, .9]);

  return extend(startState, { armToExpectedReward : { 0:p0, 1:p1}  });
} });


var params = { utility: utility, priorBelief: priorBelief };
var agent = makeAgent(params);
var trajectory = simulate(startState, agent);


print('Number of trials: ' + numberTrials);
print('Arms pulled: ' +  map(second, trajectory));
~~~~



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

