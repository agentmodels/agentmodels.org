---
layout: chapter
title: "POMDPs and Agents Who Learn From Observation"
description: Mathematical framework, implementation in WebPPL, Gridworld and restaurants example, bandit problems.
---


 
## Introduction: Learning about the world from observation

The previous chapters included MDPs where the transition function is *stochastic*. This means the agent is *uncertain* about the result of taking an action in a given state. For example in Gridworld Hiking, Alice is uncertain whether she would fall down the hill if she takes the shortcut. In an MDP the agent's uncertainty cannot be altered by observation. Transitions occur according to a particular probability distribution that is fixed (with no learnable parameters). An MDP is like a fair lottery: observing the winning ticket one week does not change the distribution on tickets the following week.

In contrast, we often face problems where our uncertainty can be *reduced* by observation. In the example of Bob choosing between restaurants, Bob would not have complete knowledge of the restaurants in his neighborhood. He'd be uncertain about opening hours, the chance of getting a table, the quality of restaurants, the exact distances between locations, and so on. This uncertainty can be reduced observation: Bob can walk to a restaurant and see whether or not it's open. In other examples, the environment is stochastic but the agent can gain knowledge of the *distribution* on outcomes. For example, in Multi-arm Bandit problems, the agent learns about the distribution over rewards given by each of the arms.

To represent decision problems where the agent's uncertainty is altered by observations, we use Partially Observable Markov Decision Processes (POMDPs). We first introduce the formalism for POMDPs and then show how to extend our agent model for MDPs to an agent model that solves POMDPs. 


## POMDP Agent Model

### Informal overview

The agent facing a POMDP has initial uncertainty about features of the environment. These could features could be external to the agent (e.g. whether a restaurant is open) or could concern the agent (e.g. the agent's position on a grid). These features influence the environment's transitions or utilities and hence are relevant to the agent's choices.

The agent does not directly observe transitions or utilities. Instead the agent learns about the unknown features of the environment *indirectly* via observations. When the agent visits a state, they receive an observation that depends on the state and their previous action (according to a fixed *observation function*). Observations can inform the agent of local, transient facts (e.g. the agent's current grid position) and of persistent features of the environment (e.g. whether a wall exists in a particular location).

In the POMDP examples we consider, the environment has the same essential MDP structure as before (save for the addition of the observation function). In an MDP, the agent is given a state and chooses an action (which causes a state transition). In a POMDP, the agent has a probability distribution over the current state. At every timestep, they (1) update this distribution on the current observation, and (2) choose an action (based on their updated distribution over states). This action causes a state transition exactly as in an MDP. 

For a concrete example, consider the Restaurant Choice Problem. Suppose Bob doesn't know whether the Noodle Shop is open. Previously, the agent's state consisted of Bob's *location* on the grid as well as the remaining time. In the POMDP case, the state also represents whether or not the Noodle Shop is open. This fact about the state determines whether Bob transitions to inside the Noodle Shop if he is adjacent. When Bob is close to the Noodle Shop, he gets to observe (via the observation function) whether or not it's open (without having to actually try it).

### Formal model

We first define a new class of decision probems (POMDPs) and then define an agent model for optimally solving these problems. Our definitions are based on refp:kaelbling1998planning. A Partially Observable Markov Decision Process (POMDP) is a tuple $$ \left\langle S,A(s),T(s,a),U(s,a),\Omega,O \right\rangle$$, where:

- $$S$$ (state space), $$A$$ (action space), $$T$$ (transition function), $$U$$ (utility or reward function) form an MDP as defined in [chapter III.1](/chapters/3a-mdp.html), with $$U$$ assumed to be deterministic. 

- $$\Omega$$ is the finite space of observations the agent can receive.

- $$O$$ is a function  $$ O\colon S \times A \to \Delta \Omega $$. This is the *observation function*, which maps an action $$a$$ and the state $$s'$$ resulting from taking $$a$$ to an observation $$o \in \Omega$$ drawn from $$O(s',a')$$.

So at each timestep, the agent transitions from state $$s$$ to state $$s' \sim T(s,a)$$ (where $$s$$ and $$s'$$ are generally unknown to the agent) having performed action $$a$$. On entering $$s'$$ the agent receives an observation $$o \sim O(s',a)$$ and a utility $$U(s,a)$$. [TODO Might be good to include influence diagram from Braziunas page 3.]. 

To characterize the behavior of an expected-utility maximizing agent, we need to formalize the belief-updating process. Let $$b$$, the current belief function, be a probability distribution over the agent's current state. Then the agent's succesor belief function $$b'$$ over their next state is the result of a Bayesian update on the observation $$o \sim O(s',a)$$ where $$a$$ is the agent's action in $$s$$.  That is:

$$
b'(s') \propto O(s',a,o)\sum_{s \in S}{T(s,a,s')b(s)}
$$

Intuitively, the probability that $$s'$$ is the new state depends on the marginal probability of transitioning to $$s'$$ (given $$b$$) and the probability of the observation $$o$$ occurring in $$s'$$. 

In our previous agent model for MDPs, we defined the expected utility of an action $$a$$ in a state $$s$$ recursively in terms of the expected utility of the resulting pair of state $$s'$$ and action $$a'$$. This same recursive characterization of expected utility still holds. The important difference is that the agent's action $$a'$$ in $$s'$$ depends on their updated belief $$b'(s')$$ given the observation they receive in $$s'$$. So the expected utility of $$a$$ in $$s$$ depends on the agent's belief $$b$$ over the state $$s$$. We call the following the *Expected Utility of State Recursion*, which defines the function $$EU_{b}$$. This is analogous to the characterization of the *value*, $$V_{b}$$, of a state relative to a belief (see p.109 in refp:kaelbling1998planning). [TODO expectation should be math style].

$$
EU_{b}[s,a] = U(s,a) + E_{s',o,a'}(EU_{b'}[s',a'_{b'}])
$$

where:

- we have $$s' \sim T(s,a)$$ and $$o \sim O(s',a)$$

- $$b'$$ is the updated belief function $$b$$ on $$o$$ (as defined above)

- $$a'_{b'}$$ is the softmax action the agent takes given belief $$b'$$

The agent cannot use the Expected Utility of State Recursion to directly compute the best action, since the agent doesn't know the state. Instead the agent takes an expectation over their belief distribution, picking the action $$a$$ that maximizes the following:

$$
EU[b,a] = E_{s \sim b}(EU_{b}[s,a])
$$

We can also represent the expected utility of action $$a$$ given belief $$b$$ in terms of a recursion on the successor belief state. We call this the *Expected Utility of Belief Recursion*. It's analogous to the Bellman update rule [add reference].

$$
EU[b,a] = E_{s \sim b}( U(s,a) + E_{s',o,a'}(EU[b',a']) )
$$

where $$s'$$, $$o$$, $$a'$$ and $$b'$$ are distributed as in the Expected Utility of State Recursion.


### Implementation of the Model
As with the agent model for MDPs, we provide a direct translation of the equations above into an agent model for solving POMDPs. The variables `nextState`, `nextObservation`, `nextBelief`, and `nextAction` correspond to $$s'$$,  $$o$$, $$b'$$ and $$a'$$ respectively, and we use the Expected Utility of Belief Recursion. The following codebox defines the `act` and `expectedUtility` functions, without defining `updateBelief`, `transition`, `observe` or `utility`. 

[TODO: can we remove the Run button. Alternative, we should do ```javascript or whatever markdown is for syntax highlighted code printing]

~~~~
var act = function(belief) {
  return Enumerate(function(){
    var action = uniformDraw(actions);
    var eu = expectedUtility(belief, action);
    factor(alpha * eu);
    return action;
  });
};

var expectedUtility = function(belief, action) {
  return expectation(
    Enumerate(function(){
      var state = sample(belief);
	  var u = utility(state, action);
	  if (state.terminateAfterAction) {
	    return u;
	  } else {
	    var nextState = transition(state, action);
	    var nextObservation = observe(nextState);
	    var nextBelief = updateBelief(belief, nextObservation, action);            
	    var nextAction = sample(act(nextBelief));   
	    return u + expectedUtility(nextBelief, nextAction);
	    }
    }));
};

// *startState* is agent's actual startState (unknown to agent)
// *priorBelief* is agent's initial belief function
var simulate = function(startState, priorBelief) {
    
  var sampleSequence = function(state, priorBelief, action) {
    var observation = observe(state);
    var belief = updateBelief(priorBelief, observation, action);
    var action = sample(act(belief));
    var output = [ [state,action] ];
      
    if (state.terminateAfterAction){
      return output;
    } else {   
      var nextState = transition(state, action);
      return output.concat(sampleSequence(nextState, belief, action));
      }
    };
  return sampleSequence(startState, priorBelief, 'startAction');
};
~~~~

## Applying the POMDP agent model

### Two-arm deterministic bandits
To illustrate the POMDP agent in action, we implement a simplified variant of the Multi-arm Bandit Problem. In this variant, there are just two arms. Pulling an arm produces a prize (deterministically). The agent does not know initially the mapping from arms to prizes but can learn by trying the arms. In our concrete example, the first arm is known to have the prize "chocolate" and the second arm either has "champagne" or has no prize at all ("nothing").  

In our implementation of this problem, we label the two arms with numbers in `[0,1]`, and use the same labels for the actions of pulling the arms. After taking action `0`, the agent transitions to a state with whatever prize is associated with `Arm0` (and gets to observe that prize). States contain properties for counting down the time (as before), as well as a `prize` property. States also contain the "latent" mapping from arms to prizes (called `armToPrize`) that determines how an agent transitions on pulling an arm.

If the agent only has one timestep in total (i.e. one bandit trial), then they will take the arm with highest expected utility (given their prior on `armToPrize`). If there are multiple trials, the agent might *explore* the lower expected utility arm (e.g. if it's maximum possible utility is higher). You should try changing the number trials to see how it affects the agent's choice on the first trial. 

~~~~

// ---------------
// Defining the Bandits decision problem

// Pull arm0 or arm1
var actions = [0,1];

// use latent "armToPrize" mapping in
// state to determine which prize agent gets
var transition = function(state, action){
  return update(state, 
                {prize: state.armToPrize[action], 
                 timeLeft: state.timeLeft - 1,
                 terminateAfterAction: state.timeLeft == 2})
};

// After pulling an arm, agent observes associated prize
var observe = function(state){return state.prize;};

var startState = { prize: 'start',
                   timeLeft:3, 
                   terminateAfterAction:false,
                   armToPrize: {0:'chocolate', 1:'champagne'}
                 };
                
// ---------------
// Defining the POMDP agent

// Agent's preferences over prizes
var utility = function(state,action){
  var prizeToUtility = {chocolate: 1, nothing: 0, champagne: 1.5, start:0};
  return prizeToUtility[state.prize];
};

// Agent's prior prior includes possibility that arm1 has no prize
// (instead of champagne)
var alternativeStartState = update(startState, {armToPrize:{0:'chocolate', 1:'nothing'}});

var priorBelief = Enumerate(function(){
  return categorical( [.5, .5], [startState, alternativeStartState]);
});


// Agent's belief update: directly translates the belief update
// equation above

var updateBelief = function(belief, observation, action){
  return Enumerate(function(){
    var state = sample(belief);
    var predictedNextState = transition(state, action);
    var predictedObservation = observe(predictedNextState);
    condition(_.isEqual(predictedObservation, observation));
    return predictedNextState;
  });
};

var act = dp.cache(
  function(belief) {
    return Enumerate(function(){
      var action = uniformDraw(actions);
      var eu = expectedUtility(belief, action);
      factor(100 * eu);
      return action;
    });
  });

var expectedUtility = dp.cache(
  function(belief, action) {
    return expectation(
      Enumerate(function(){
	var state = sample(belief);
	var u = utility(state, action);
	if (state.terminateAfterAction) {
	  return u;
	} else {
	  var nextState = transition(state, action);
	  var nextObservation = observe(nextState);
	  var nextBelief = updateBelief(belief, nextObservation, action);            
	  var nextAction = sample(act(nextBelief));   
	  return u + expectedUtility(nextBelief, nextAction);
	}
      }));
  });


var simulate = function(startState, priorBelief) {
    
  var sampleSequence = function(state, priorBelief, action) {
    var observation = observe(state);
    var belief = action=='startAction' ? priorBelief : updateBelief(priorBelief, observation, action);
    var action = sample(act(belief));
    var output = [ [state,action] ];
         
    if (state.terminateAfterAction){
      return output;
    } else {   
      var nextState = transition(state, action);
      return output.concat(sampleSequence(nextState, belief, action));
    }
  };
  return sampleSequence(startState, priorBelief, 'startAction');
};



var displayTrajectory = function( trajectory ){
  var out = map( function(state_action){
    var previousPrize = state_action[0].prize;
    var nextAction = state_action[1];
    return [previousPrize, nextAction];
  }, trajectory);  
  var out = _.flatten(out);
  return out.slice(1,out.length-1);
};

displayTrajectory(simulate(startState, priorBelief));
~~~~

### Bandits with stochastic observations
The bandit problem above is especially simple because pulling an arm *deterministically* results in a prize (which the agent directly observes). So there is a fixed, finite number of beliefs about the `armToPrize` mapping that the agent can have. This number depends on the number of arms but not on the number of trials. [TODO: exact complexity of the two-arm case? Show our code achieves it.]

We can generalize this bandit problem to the more standard *stochastic* multi-arm bandits. In this case, pulling an arm yields a distribution on prizes and the agent does not know the distribution. In the example below, we suppose that there are only two prizes "zero" and "one" which yield utilities 0 and 1. Each arm $$i$$ yields the prize "one" with probability $$p_i$$ and "zero" with probability $$1-p_i$$. This is known as *binary* or *Bernoulli* bandits and has been studied extensively (refp:kaelbling1996reinforcement -- kaelbling, littman, moore 1996 reinforcement learning). In this problem, the number of possible beliefs about $$p_i$$ will increase with the number of trials. More generally, this problem takes time exponential in the number of trials. [Show our code has this property -- maybe add some more detail or references.]

[TODO:
- For this codebox and the ones for scaling, I'd like the code to be short and simple -- basically just for viewing the results and varying the input parameters. So the code should construct the bandit world, the agent prior and then do the output / graph visualization. So you should try to move every helper function with *manifestState* or *loc* in it to stochasticBandits.wppl. For instance, you can have a `buildStartState(timeLeft,armToERP)`, `getStochasticBanditPrior(startState, armToPrior)`, `getUtilityFunction`, `displayBanditTrajectory`. The names should be specific to stochastic bandits -- so pick whatever seems sensible.

We could also put these helper functions above the 'fold', following DIPPL:
http://dippl.org/chapters/05-particlefilter.html

- Part of the slowness is presumably that we don't have fastUpdate for stochastic bandits. But we could write a special version.

- For the first codebox, have `displayTrajectory` print out each result of pulling an arm on a newline. Something like: '\n Arm: ' + arm + ' Reward: ' + state', or mb something more like a table with columbs (where you just separate the arm and the reward with some space or a '|' character).


- For displaying runtime, it'd be good to convert to seconds. 

~~~~
var world = makeStochasticBanditWorld(2);

var probablyChampagneERP = categoricalERP([0.2, 0.8], ['nothing', 'champagne']);

var probablyNothingERP = categoricalERP([0.8, 0.2], ['nothing', 'champagne']);

var startState = {manifestState: {loc: 'start',
				                  timeLeft: 10,
								  terminateAfterAction: false},
	              latentState: {0: deltaERP('chocolate'),
				                1: probablyChampagneERP}};

var prior = Enumerate(function(){
  var manifestState = startState.manifestState;
  var latentState = {0: deltaERP('chocolate'),
		             1: uniformDraw([probablyNothingERP,
				                     probablyChampagneERP])};
  return buildState(manifestState, latentState);
});

var prizeToUtility = {start: 0,
	                  nothing: 0,
					  chocolate: 1,
					  champagne: 1.5};

var utility = function(state,action){
  var loc = state.manifestState.loc;
  return prizeToUtility[loc];
};

var agentParams = {utility: utility,
		           alpha: 10000,
		           priorBelief: prior,
		           fastUpdateBelief: false};

var agent = makeBeliefAgent(agentParams, world);

var displayTrajectory = function(trajectory) {
  var getLocAction = function(stateAction) {
    var state = stateAction[0];
    var action = stateAction[1];
    return [state.manifestState.loc, action];
  };

  var locsActions = map(getLocAction, trajectory);
  var flatLocsActions = _.flatten(locsActions);
  return flatLocsActions.slice(1, flatLocsActions.length - 1);
};

var trajectory = simulateBeliefAgent(startState, world, agent, 'stateAction');

displayTrajectory(trajectory);

~~~~

Scaling:

~~~~

// test to show the scaling properties of stochastic bandits

var varyTime = function(n) {

  var world = makeStochasticBanditWorld(2);

  var probablyChampagneERP = categoricalERP([0.2, 0.8], ['nothing', 'champagne']);

  var probablyNothingERP = categoricalERP([0.8, 0.2], ['nothing', 'champagne']);

  var startState = {manifestState: {loc: 'start',
				                    timeLeft: n,
									terminateAfterAction: false},
		            latentState: {0: deltaERP('chocolate'),
				                  1: probablyChampagneERP}};

  var prior = Enumerate(function(){
    var manifestState = startState.manifestState;
    var latentState = {0: deltaERP('chocolate'),
		               1: uniformDraw([probablyNothingERP,
				                       probablyChampagneERP])};
    return buildState(manifestState, latentState);
  });

  var prizeToUtility = {start: 0,
			            nothing: 0,
						chocolate: 1,
						champagne: 1.5};

  var utility = function(state,action){
    var loc = state.manifestState.loc;
    return prizeToUtility[loc];
  };

  var agentParams = {utility: utility,
	                 alpha: 100,
		             priorBelief: prior,
		             fastUpdateBelief: false};

  var agent = makeBeliefAgent(agentParams, world);

  var displayTrajectory = function(trajectory) {
    var getLocAction = function(stateAction) {
      var state = stateAction[0];
      var action = stateAction[1];
      return [state.manifestState.loc, action];
    };

    var locsActions = map(getLocAction, trajectory);
    var flatLocsActions = _.flatten(locsActions);
    return flatLocsActions.slice(1, flatLocsActions.length - 1);
  };

  var f = function() {
    var trajectory = simulateBeliefAgent(startState, world, agent, 'stateAction');
    return displayTrajectory(trajectory);
  };

  return timeit(f).runtimeInMilliseconds.toPrecision(3);
};

// Varying the lifetime of the agent
var lifetimes = _.range(16).slice(2);
var runtimes = map(varyTime, lifetimes);

print('Runtime in ms for lifetimes ' + lifetimes + '\n' + runtimes);

viz.bar(lifetimes, runtimes);

// note: this takes approximately 30 seconds to run
~~~~

~~~~

// test to show the scaling properties of stochastic bandits

var varyArms = function(n) {

  var world = makeStochasticBanditWorld(n);

  var probablyChampagneERP = categoricalERP([0.2, 0.8], ['nothing', 'champagne']);

  var probablyNothingERP = categoricalERP([0.8, 0.2], ['nothing', 'champagne']);
  
  var makeLatentState = function(numArms) {
    return map(function(x){return probablyChampagneERP;}, _.range(numArms));
  };

  var startState = {manifestState: {loc: 'start',
				                    timeLeft: 5,
									terminateAfterAction: false},
		            latentState: makeLatentState(n)};

  var latentSampler = function(numArms) {
    return map(function(x){return uniformDraw([probablyNothingERP,
					                           probablyChampagneERP]);},
	           _.range(numArms));
  };

  var prior = Enumerate(function(){
    var manifestState = startState.manifestState;
    var latentState = latentSampler(n);
    return buildState(manifestState, latentState);
  });

  var prizeToUtility = {start: 0,
			            nothing: 0,
						chocolate: 1,
						champagne: 1.5};

  var utility = function(state,action){
    var loc = state.manifestState.loc;
    return prizeToUtility[loc];
  };

  var agentParams = {utility: utility,
		             alpha: 100,
		             priorBelief: prior,
		             fastUpdateBelief: false};

  var agent = makeBeliefAgent(agentParams, world);
  
  var displayTrajectory = function(trajectory) {
    var getLocAction = function(stateAction) {
      var state = stateAction[0];
      var action = stateAction[1];
      return [state.manifestState.loc, action];
    };

    var locsActions = map(getLocAction, trajectory);
    var flatLocsActions = _.flatten(locsActions);
    return flatLocsActions.slice(1, flatLocsActions.length - 1);
  };

  var f = function() {
    var trajectory = simulateBeliefAgent(startState, world, agent, 'stateAction');
    return displayTrajectory(trajectory);
  };

  return timeit(f).runtimeInMilliseconds.toPrecision(3);

};

// varying the number of arms

var arms = [1,2,3];
var runtimes = map(varyArms, arms);

print('Runtime in ms for arms ' + arms + '\n' + runtimes);

viz.bar(arms, runtimes);

// note the increase in order of magnitude. If we called this for 4 arms, the
// increase would continue, meaning that it would take impractically long.

~~~~


### Gridworld with observations
As we discussed above, an agent in the Restaurant Choice problem is likely to be uncertain about some features of the environment. We consider a variant of the Restaurant Choice problem where the agent is uncertain about which restaurants are open. The agent can observe whether a restaurant is open by moving to a square on the grid adjacent to the restaurant. If the restaurant is open, the agent can enter it (and receive utility).

In this POMDP version of Restaurant Choice, a rational agent can exhibit behavior that never occurs in the MDP version. First, suppose the agent has a prior belief that Donut South is likely to be closed and Donut North is likely to be open. Then the agent might go to Donut North despite Donut South being closer (see example below). Second, suppose the agent believes the Noodle Shop is likely open when it's actually closed. Then the agent might go to Noodle Shop, see it's closed and then take the long loop round to Vegetarian Cafe (which would not make sense if the Noodle Shop was known to be closed from the start). This is shown in the second example below. 

[Add these two examples using library functions]. 

### Possible additions
- Doing belief update online vs belief doing a batch update every time. Latter is good if belief updates are rare and if we are doing approximate inference (otherwise the errors in approximations will compound in some way). Maintaining observations is also good if your ability to do good approximate inference changes over time. (Or least maintaining compressed observations or some kind of compressed summary statistic of the observation -- e.g. .jpg or mp3 form). This is related to UDT vs CDT and possibly to the episodic vs. declarative memory in human psychology. [Add a different *updateBelief* function to illustrate.]

--------------

[Table of Contents](/)
