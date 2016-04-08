---
layout: chapter
title: Bounded Agents-- Greedy and Myopic
description: Fast approximate planning algorithms that assume a short time horizon for utilities (Greedy) or for obserations (Myopic)

---

### Introduction
In the previous chapter, we extended our earlier agent model for solving MDPs optimally to a model of planning for hyperbolic discounters. The goal was to better capture human behavior by incorporating one of the most prominent and well studied human *biases*. As we discussed [earlier](/chapters/5-biases-intro), any bounded agent will be unable to solve certain computational problems optimally. So when modeling human behavior (e.g. for Inverse Reinforcement Learning), we might produce better generative models by incorporating planning algorithms that are sub-optimal but which perform well given human computational bounds (e.g. they might be "resource rational" in the sense of CITE). This chapter describes two approximate planning algorithms that are conceptually simple and much more scalable than optimal planning: Greedy planning and Myopic Exploration. Each can be implemented by adding a few lines of code to the optimal POMDP agent. 

### Greedy Planning: the basic idea
One reason optimal planning is computationally difficult is that it chooses actions in a way that takes into account the entire future. The (PO)MDP agent we described previously reasons backwards from utility of (belief) states at the final timestep. It considers actions on earlier timesteps based on whether they lead to good final states. (With an infinite time horizon, an optimal agent must consider the expected utility of being in every possible state, including states only reachable after a very long duration). Suppose we used this approach when playing against a predictable chess algorithm for a fixed and very large number of timesteps. Then it would simulate every possible chess game up to that duration, including many games that are incredibly unlikely.

The obvious alternative to taking into account the entire future when taking an action is to consider only the short term. For example, in a problem that lasts $$1000$$ timesteps, you take your first action by optimizing for the first $$10$$ timesteps. For your second action, you recompute your plan, optimizing for timesteps $$2-11$$, and so on. Whereas the optimal agent computes a complete plan or *policy* before the first timestep and does no more computation after that, the "greedy" alternative involves computing a short-term plan at every timestep. Spreading out the computation in this way can be much more tractable than the optimal approach. The Greedy approach only considers states or belief-states that it actually enters or that it gets close to, while the Optimal approach considers every possible state or belief-state.

The Greedy approach will work best if continually optimizing for the short-term produces good results in the long-term. It's easy to imagine probems where this doesn't hold: e.g. if there's a big prize at the end of a long corridor (with no rewards along the way). A remedy is to find a proxy for the long-term expected utility of being in a state that is cheap to compute. For example, you can Greedily optimize the score of future chess positions using a cheaply computed evaluation function. In this chapter, we only consider the simplest kind of Greedy planning, where the agent optimizes for short-term utility (rather than for some tractable proxy of long-term utility).


### Greedy Planning: implementation and examples
We consider the simplest kind of Greedy agent. This agent takes the expected-utility maximizing action, assuming that the decision problem ends $$C_g$$ steps into the future. The cutoff or *bound* $$C_g > 0$$ will typically be much smaller than the time horizon for the decision problem. You will notice the similarity between the Greedy agent and the hyperbolic discounting agents. Both agents make plans based (mostly) on short-term rewards. Both agents revise these plans at every timestep. And the Naive and Greedy agents both have implicit models of their future selves that are incorrect. A major difference is that Greedy planning is much easier to make computationally fast. Given their similarity, it is unsurprising that we implement the Greedy agent using the concept of *delay* described in the previous chapter. The formalization and implementation of the Greedy agent is left as an exercise.

>**Exercise:** Formalize POMDP and MDP versions of the Greedy agent by modifiying the equations for the expected 





### Myopic Exploration: the basic idea
In Chapter [POMDPs], we noted that solving a finite-horizon Multi-arm bandit problem is intractable in the number of arms and trials. So bounded agents will use some sub-optimal but tractable algorithm for this problem. In this chapter we describe and implement a widely-studied approach to Bandits (and POMDPs generally) that is sub-optimal but which can be very effective in practice. We refer to the approach as *Myopic Exploration*, because it is "myopic" or "greedy" with respect to exploration. The idea is that the agent at time $$t_0$$ assumes they can only *explore* (i.e. update beliefs from observations) up to some cutoff point $$C$$ steps into the future. After that point they just *exploit* (i.e. they gain rewards but don't update from the rewards they observe). In fact, the agent will continue to update beliefs after time $$t_0+C$$; like the Naive hyperbolic discounter the myopic agent has an incorrect model of its future self. We call an agent that uses Myopic Exploration a "Myopic Agent". This will be precisely defined below. 

Myopic Exploration is not only useful for solving Bandit problems efficiently, but also provides a good fit to human performance in Bandit problems. In what follows, we describe Myopic Exploration in more detail, explain how to incorporate it into out POMDP agent model, and then exhibit its performance on Bandit problems.

### Myopic Exploration: applications and limitations
As noted above, Myopic Exploration has been studied in Machine Learning refp:gonzalez2015glasses and Operations Research refp:ryzhov2012knowledge as part of algorithms for generalized Bandit problems. In most cases, the cutoff point $$C$$ after which the agent assumes himself to exploit is set to $$C=1$$. This results in a scalable, analytically tractable optimization problem: pull the arm that maximizes the expected value of future exploitation given you pulled that arm. This "future exploitation" means that you pick the arm that is best in expectation for the rest of time. The Myopic Agent with $$C=1$$ has also been successfully used a model of human play in Bandit problems refp:zhang2013forgetful. 

We've presented Bandit problems with a finite number of arms, and with discrete rewards that are uncorrelated across arms. Myopic Exploration works well in this setting but also works for generalized Bandit Problems: e.g. when rewards are correlated, when rewards are continuous, and in the "Bayesian Optimization" setting where instead of a fixed number of arms the goal is to optimize high-dimensional real-valued function refp:ryzhov2012knowledge. 

Myopic Exploration will not work well for POMDPs in general. Suppose I'm looking for a good restaurant in a foreign city. A good strategy is to walk to a busy street and then find the restaurant with the longest line. If reaching the busy street takes longer than the myopic cutoff $$C$$, then a Myopic agent would not model himself as learning which restaurant has the longest line -- and hence would not recognize this as a good strategy. The Myopic agent would only carry out this strategy if it could observe the restaurants before the cutoff $$C$$. (This kind of POMDP could easily be represented in our Gridworld framework.). This highlights a way in which Bandit problems are distinctive from general POMDPs. In a Bandit problem, you can always explore every arm: you never need to first move to an appropriate state. So even the Myopic Agent with $$C=1$$ compares the information value of every possible observation that the POMDP can yield.

The Myopic agent has an incorrect model of his future self, assuming his future self stops updating after cutoff point $$C$$. This incorrect "self-modeling" is also a property of well-known model-free RL agents. For example, a Q-learner's estimation of expected utilities for states ignores the fact that the Q-learner will randomly explore with some probability. SARSA, on the other hand, does take its random exploration into account when computing this estimate. But it doesn't model the way in which its future exploration behavior will make certain actions useful in the present (as in the example of finding a restaurant in a foreign city).

### Myopic Exploration: formal model
Myopic Exploration only makes sense in the context of an agent that is capable of learning from observations (i.e. in the POMDP rather than MDP setting). So our goal is to generalize our agent model for solving POMDPs to a Myopic Exploration with $$C \in [1,\infty]$$.

**Exercise:** Before reading on, modify the equations defining the [POMDP agent](/chapters/3c-pomdp) in order to generalize the agent model to include Myopic Exploration. The optimal POMDP agent will be the special case when $$C=\infty$$.

------------

To extend the POMDP agent to the Myopic agent, we use the idea of *delays* from the previous chapter. These delays are not used to evaluate future rewards (as any discounting agent would use them). They are used to determine how future actions are simulated. If the future action occurs when delay $$d$$ exceeds cutoff point $$C$$, then the simulated future self does not do a belief update before taking the action. (This makes the Myopic agent analogous to the Naive agent: both simulate the future action by projecting the wrong delay value onto their future self). 

We retain the notation from the definition of the POMDP agent and skip directly to the equation for the expected utility of a state, which we modify for the Myopic agent with cutoff point $$C \in [1,\infty]$$:

$$
EU_{b}[s,a,d] = U(s,a) + \mathbb{E}_{s',o,a'}(EU_{b'}[s',a'_{b'},d+1])
$$

where:

- $$s' \sim T(s,a)$$ and $$o \sim O(s',a)$$

- $$a'_{b'}$$ is the softmax action the agent takes given new belief $$b'$$

- the new belief state $$b'$$ is defined as:

$$
b'(s') \propto I_C(s',a,o,d)\sum_{s \in S}{T(s,a,s')b(s)}
$$

<!-- problem with < sign in latex math-->
where $$I_C(s',a,o,d) = O(s',a,o)$$ if $$d$$ < $$C$$ and $$I_C(s',a,o,d) = 1$$ otherwise.

The key part is the definition of $$b'$$. The Myopic agent assumes his future self updates only on his last action $$a$$ and not on observation $$o$$. So the future self will know about state changes that follow a priori from his actions. (In a deterministic Gridworld, the future self would know his new location and that the time remaining had been counted down).

The implementation of the Myopic agent in WebPPL is a direct translation of the definition provided above.

**Exercise:** Modify the code for the POMDP agent [todo link to codebox] to represent a Myopic agent.


### Myopic Exploration for Bandits and Gridworld

We show the performance of the Myopic agent on Multi-Arm bandits.

For 2-arms, Myopic with D=1 is optimal. Verify this and compare runtime.

~~~~
var world = makeStochasticBanditWorld(2);
var worldObserve = world.observe;
var observe = getFullObserve(worldObserve);
var transition = world.transition;

var probablyChampagneERP = categoricalERP([0.4, 0.6], ['nothing', 'champagne']);
var probablyNothingERP = categoricalERP([0.6, 0.4], ['nothing', 'champagne']);

var trueLatent = {0: probablyNothingERP,
		          1: probablyChampagneERP};

var timeLeft = 7;

var startState = buildStochasticBanditStartState(timeLeft, trueLatent);

var prior = Enumerate(function(){
  var latentState = {0: uniformDraw([probablyChampagneERP, probablyNothingERP]),
		             1: categorical([0.6, 0.4], [probablyChampagneERP,
					                        	 probablyNothingERP])};
  return buildStochasticBanditStartState(timeLeft, latentState);
});

var prizeToUtility = {start: 0, nothing: 0, champagne: 2};
var utility = makeStochasticBanditUtility(prizeToUtility);

var optimalAgentParams = {utility: utility,
			              alpha: 100,
						  priorBelief: prior,
						  fastUpdateBelief: false};
var optimalAgent = makeBeliefAgent(optimalAgentParams, world);

var myopicAgentParams = {utility: utility,
			             alpha: 100,
						 priorBelief: prior,
						 sophisticatedOrNaive: 'naive',
						 boundVOI: {on: true, bound: 1},
						 noDelays: false,
						 discount: 0,
						 myopia: {on: false, bound: 0},
						 fastUpdateBelief: false};
var myopicAgent = makeBeliefDelayAgent(myopicAgentParams, world);

var nearlyEqualActionERPs = function(erp1, erp2) {
  var nearlyEqual = function(float1, float2) {
    return Math.abs(float1 - float2) < 0.05;
  };
  return nearlyEqual(erp1.score([], 0), erp2.score([], 0))
    && nearlyEqual(erp1.score([], 1), erp2.score([], 1));
};

// it's important that we simulate the two agents such that they get the same
// prizes when pulling the same arms, so that we can check if their
// actions are the same. We could not ensure this by simply simulating one agent
// and then the other.
var sampleTwoSequences = function(states, priorBeliefs, actions) {
///fold:
  var optimalState = states[0];
  var optimalPriorBelief = priorBeliefs[0];
  var optimalAction = actions[0];

  var optimalAct = optimalAgent.act;
  var optimalUpdateBelief = optimalAgent.updateBelief;
  
  var myopicState = states[1];
  var myopicPriorBelief = priorBeliefs[1];
  var myopicAction = actions[1];
  
  var myopicAct = myopicAgent.act;
  var myopicUpdateBelief = myopicAgent.updateBelief;

  var optimalObservation = observe(optimalState);
  var myopicObservation = observe(myopicState);
  
  var delay = 0;
  var newMyopicBelief = myopicUpdateBelief(myopicPriorBelief, myopicObservation,
					                       myopicAction, delay);
  var newOptimalBelief = optimalUpdateBelief(optimalPriorBelief,
					                         optimalObservation, optimalAction);

  var newMyopicActionERP = myopicAct(newMyopicBelief, delay);
  var newOptimalActionERP = optimalAct(newOptimalBelief);

  var newMyopicAction = sample(newMyopicActionERP);
  // if ERPs over actions are almost the same, have the agents pick the same
  // action
  var newOptimalAction = nearlyEqualActionERPs(newMyopicActionERP,
					       newOptimalActionERP)
	? newMyopicAction : sample(newOptimalActionERP);

  var optimalLocAction = [optimalState.manifestState.loc, newOptimalAction];
  var myopicLocAction = [myopicState.manifestState.loc, newMyopicAction];

  var output = [optimalLocAction, myopicLocAction];

  if (optimalState.manifestState.terminateAfterAction) {
    return output;
  } else {
    var nextPriorBeliefs = [newOptimalBelief, newMyopicBelief];
    var nextActions = [newOptimalAction, newMyopicAction];
    if (_.isEqual(optimalState, myopicState) && _.isEqual(newOptimalAction,
							                              newMyopicAction)) {
      // if actions are the same, transition to the same state
      var nextState = transition(optimalState, newOptimalAction);
      var nextStates = [nextState, nextState];
      var recurse = sampleTwoSequences(nextStates, nextPriorBeliefs,
			                	       nextActions);
      return [optimalLocAction.concat(recurse[0]),
	          myopicLocAction.concat(recurse[1])];
    } else {
      var nextOptimalState = transition(optimalState, newOptimalAction);
      var nextMyopicState = transition(myopicState, newMyopicAction);
      var nextStates = [nextOptimalState, nextMyopicState];
      var recurse = sampleTwoSequences(nextStates, nextPriorBeliefs,
			                	       nextActions);
      return [optimalLocAction.concat(recurse[0]),
	          myopicLocAction.concat(recurse[1])];
    }
  }
///	
};

var startAction = 'noAction';

var trajectories = sampleTwoSequences([startState, startState], [prior, prior],
                                      [startAction, startAction]);
var length = trajectories[0].length;
print('Trajectory of optimal agent: ' + trajectories[0].slice(1, length - 1));
print('Trajectory of myopic agent: ' + trajectories[1].slice(1, length - 1));
~~~~

Scaling of myopic agent:

~~~~
var varyTime = function(n) {
  var world = makeStochasticBanditWorld(2);

  var probablyChampagneERP = categoricalERP([0.2, 0.8], ['nothing', 'champagne']);
  var probablyNothingERP = categoricalERP([0.8, 0.2], ['nothing', 'champagne']);

  var trueLatent = {0: deltaERP('chocolate'),
  		            1: probablyChampagneERP};
  var falseLatent = update(trueLatent, {1: probablyNothingERP});

  var startState = buildStochasticBanditStartState(n, trueLatent);

  var prior = Enumerate(function(){
    var latent = uniformDraw([trueLatent, falseLatent]);
    return buildStochasticBanditStartState(n, latent);
  });

  var prizeToUtility = {start: 0, nothing: 0, chocolate: 1, champagne: 1.5};
  var utility = makeStochasticBanditUtility(prizeToUtility);

  var agentParams = {utility: utility,
	                 alpha: 100,
					 priorBelief: prior,
					 sophisticatedOrNaive: 'naive',
					 boundVOI: {on: true, bound: 1},
					 noDelays: false,
					 discount: 0,
					 myopia: {on: false, bound: 0},
					 fastUpdateBelief: false};
  var agent = makeBeliefDelayAgent(agentParams, world);

  var f = function() {
    return simulateBeliefDelayAgent(startState, world, agent, 'stateAction');
  };

  return timeit(f).runtimeInMilliseconds.toPrecision(3) * 0.001;
};

var lifetimes = _.range(16).slice(2);
var runtimes = map(varyTime, lifetimes);

print('Runtime in sec for lifetimes ' + lifetimes + '\n' + runtimes);

viz.line(lifetimes, runtimes);
~~~~

~~~~
var varyArms = function(n) {
  var world = makeStochasticBanditWorld(n);

  var probablyChampagneERP = categoricalERP([0.2, 0.8], ['nothing', 'champagne']);
  var probablyNothingERP = categoricalERP([0.8, 0.2], ['nothing', 'champagne']);
  
  var makeLatentState = function(numArms) {
    return map(function(x){return probablyChampagneERP;}, _.range(numArms));
  };

  var startState = buildStochasticBanditStartState(4, makeLatentState(n));

  var latentSampler = function(numArms) {
    return map(function(x){return uniformDraw([probablyNothingERP,
                    					       probablyChampagneERP]);},
	           _.range(numArms));
  };
  var prior = Enumerate(function(){
    var latentState = latentSampler(n);
    return buildStochasticBanditStartState(4, latentState);
  });

  var prizeToUtility = {start: 0, nothing: 0, champagne: 1};

  var utility = makeStochasticBanditUtility(prizeToUtility);
  var agentParams = {utility: utility,
		             alpha: 100,
					 priorBelief: prior,
					 sophisticatedOrNaive: 'naive',
					 boundVOI: {on: true, bound: 1},
					 noDelays: false,
					 discount: 0,
					 myopia: {on: false, bound: 0},
					 fastUpdateBelief: false};
  var agent = makeBeliefDelayAgent(agentParams, world);

  var f = function() {
    var trajectory = simulateBeliefDelayAgent(startState, world, agent, 'stateAction');
    return trajectory;
  };

  return timeit(f).runtimeInMilliseconds.toPrecision(3) * 0.001;

};

var arms = [1,2,3];
var runtimes = map(varyArms, arms);

print('Runtime in sec for arms ' + arms + '\n' + runtimes);

viz.bar(arms, runtimes);
~~~~

For >2 arms, I believe Myopic D=1 is not optimal. Verify this. It should be much faster as the number of arms grows. (One easy way to speed it up is to have a special *updateBelief* in *beliefDelayAgent* for stochastic bandits. the only difference is that once delay>=C, you should just directly update the timeLeft, assuming you have belief in *ERPOverLatentState*. This will avoid the Enumerate for *nextBelief*.)

Probably: we should have an example of the agent that's myopic in utilities. We'd like examples that distinguish it from both the optimal agent and the boundVOI agent. 

TODO
We make a Gridworld version of the "Restaurant Search" problem. The agent is uncertain of the quality of all of the restaurants and has an independent uniform prior on each one, in particular `uniformDraw( _.range(1,11) )'. By moving adjacent to a restaurant, the agent observes the quality (e.g. by seeing how full the restaurant is or how good it looks from the menu). An image of the grid, which includes the true latent restaurant utilities and disiderata for where the agent should end up is in: /assets/img/5b-myopia-gridworld.png.

Cell references are spreadsheet-style: [A, B, C, ... ] for columns and [1,2,3 ...] for rows. 

![myopia gridworld](/assets/img/5b-myopia-gridworld.png)

Assuming we want to stick with "no uncertainty over utilities" and "utilities depend only on state", we would have to implement this by having extra states associated with the utility values in range(1,11). The latent state is the table {restaurantA:utilityRestaurantA}. The transition function is the normal gridworld transition, with an extra condition s.t. when the agent goes to a restaurant they get sent to state corresponding to the restaurant's utility. (Whatever solution is used need not be general. We don't need to show the code, we just need to make the example work).


~~~~
// optimal_agent_restaurant_search
var gridworld = makeRestaurantSearchMDP({noReverse: true});
var world = makeBanditGridworld(gridworld);
var feature = world.feature;
var startState = restaurantSearchStartState;

var agentPrior = Enumerate(function(){
  var rewardE = flip() ? 5 : 0;
  var latentState = {A: 3,
		             B: uniformDraw(range(6)),
		             C: uniformDraw(range(6)),
		             D: 5 - rewardE,
		             E: rewardE};
  return buildState(startState.manifestState, latentState);
});

var params = {
  utility: makeBanditGridworldUtility(feature, -0.01),
  alpha: 1000,
  priorBelief: agentPrior
};

var agent = makeBeliefAgent(params, world);

var trajectory = simulateBeliefAgent(startState, world, agent, 'stateAction');

GridWorld.draw(world, {trajectory: trajectory})

~~~~




