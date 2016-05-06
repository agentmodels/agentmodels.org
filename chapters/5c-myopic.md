---
layout: chapter
title: Bounded Agents — Greedy and Myopic
description: Fast approximate planning algorithms that assume a short time horizon for utilities (Greedy) or for obserations (Myopic)

---

### Introduction
In the previous chapter, we extended our earlier agent model for solving MDPs optimally to a model of planning for hyperbolic discounters. The goal was to better capture human behavior by incorporating one of the most prominent and well studied human *biases*. As we discussed [earlier](/chapters/5-biases-intro), any bounded agent will be unable to solve certain computational problems optimally. So when modeling human behavior (e.g. for Inverse Reinforcement Learning), we might produce better generative models by incorporating planning algorithms that are sub-optimal but which perform well given human computational bounds (e.g. they might be "resource rational" in the sense of CITE). This chapter describes two approximate planning algorithms that are conceptually simple and much more scalable than optimal planning: Greedy planning and Myopic Exploration. Each can be implemented by adding a few lines of code to the optimal POMDP agent. 

## Greedy Planning: the basic idea
One reason optimal planning is computationally difficult is that it chooses actions in a way that takes into account the entire future. The (PO)MDP agent we described previously reasons backwards from utility of (belief) states at the final timestep. It considers actions on earlier timesteps based on whether they lead to good final states. (With an infinite time horizon, an optimal agent must consider the expected utility of being in every possible state, including states only reachable after a very long duration). Suppose we used this approach when playing against a predictable chess algorithm for a fixed and very large number of timesteps. Then it would simulate every possible chess game up to that duration, including many games that are incredibly unlikely.

The obvious alternative to taking into account the entire future when taking an action is to consider only the short term. For example, in a problem that lasts 1000 timesteps, you take your first action by optimizing for the first 10 timesteps. For your second action, you recompute your plan, optimizing for timesteps 2-11, and so on. Whereas the optimal agent computes a complete plan or *policy* before the first timestep and does no more computation after that, the "greedy" alternative involves computing a short-term plan at every timestep. Spreading out the computation in this way can be much more tractable than the optimal approach. The Greedy approach only considers states or belief-states that it actually enters or that it gets close to, while the Optimal approach considers every possible state or belief-state.

The Greedy approach will work best if continually optimizing for the short-term produces good results in the long-term. It's easy to imagine probems where this doesn't hold: e.g. if there's a big prize at the end of a long corridor (with no rewards along the way). A remedy is to find a proxy for the long-term expected utility of being in a state that is cheap to compute. For example, you can Greedily optimize the score of future chess positions using a cheaply computed evaluation function. In this chapter, we only consider the simplest kind of Greedy planning, where the agent optimizes for short-term utility (rather than for some tractable proxy of long-term utility).


### Greedy Planning: implementation and examples
We consider the simplest kind of Greedy agent. This agent takes the expected-utility maximizing action, assuming that the decision problem ends $$C_g$$ steps into the future. The cutoff or *bound* $$C_g > 0$$ will typically be much smaller than the time horizon for the decision problem.

You will notice the similarity between the Greedy agent and the hyperbolic discounting agents. Both agents make plans based (mostly) on short-term rewards. Both agents revise these plans at every timestep. And the Naive and Greedy agents both have implicit models of their future selves that are incorrect. A major difference is that Greedy planning is much easier to make computationally fast. Given their similarity, it is unsurprising that we implement the Greedy agent using the concept of *delay* described in the previous chapter. The formalization and implementation of the Greedy agent is left as an exercise.

>**Exercise:** Formalize POMDP and MDP versions of the Greedy agent by modifiying the equations for the expected utility of state-action pairs or belief-state-action pairs. Implement the agent by modifying the code for the POMDP and MDP agents. Verify that the agent behaves sub-optimally (but more efficiently) on Gridworld and Bandit problems. 

------

The Greedy agent will do well if good short-term actions produce good long-term consequences. In Bandit problems, elaborate long-terms plans are not needed to reach particular desirable future states. It turns out that a maximally Greedy agent, who only cares about the immediate reward ($$C_g = 1$$), does well on the standard Multi-arm bandit problem -- provided that it has some noise in its actions TODO add sutton barto cite refp:sutton119xreinforcement.

The next codeboxes show the performance of the Greedy agent on Bandit problems. The first codebox is a two-arm Bandit problem, illustrated in Figure 1. We use a Greedy agent with high softmax noise: $$C_g=1$$ and $$\alpha=10$$. The Greedy agent's average reward over 100 trials is close to the expected average reward given perfect knowledge of the arms.

<img src="/assets/img/5b-greedy-bandit.png" alt="diagram" style="width: 600px;"/>

>**Figure 1:** Bandit problem. The curly brackets contain possible probabilities according to the agent's prior (the bolded number is the true probability). For `arm0`, the agent has a uniform prior on the values $$\{0, 0.25, 0.5, 0.75, 1\}$$ for the probability the arm yields the reward 1.5.

<br>

~~~~
// noisy_greedy_regret_ratio

// Construct world: One bad arm, one good arm, 100 trials. 

var trueArmToPrizeERP = {
  0: categoricalERP([0.25, 0.75], [1.5, 0] ),
  1: categoricalERP([0.5, 0.5], [1, 0])
};
var numberOfTrials = 100;
var bandit = makeBandit({
  numberOfTrials: numberOfTrials,
  numberOfArms: 2,
  armToPrizeERP: trueArmToPrizeERP,
  numericalPrizes: true
});
var world = bandit.world;
var startState = bandit.startState;


// Construct greedy agent

// Arm0 is a mixture of [0,1.5] and Arm1 of [0,1]
var agentPrior = Enumerate(function(){
  var prob15 = uniformDraw([0, 0.25, 0.5, 0.75, 1]);
  var prob1 = uniformDraw([0, 0.25, 0.5, 0.75, 1]);
  var armToPrizeERP = {0: categoricalERP([prob15, 1 - prob15], [1.5, 0]),
		               1: categoricalERP([prob1, 1 - prob1], [1, 0])};
  return makeBanditStartState(numberOfTrials, armToPrizeERP);
});

var greedyBound = 1;
var alpha = 10; // noise level

var params = {
  alpha: alpha,
  priorBelief: agentPrior,
  myopia: {on: true, bound: greedyBound},
  boundVOI: {on: false, bound: 0},
  noDelays: false,
  discount: 0,
  sophisticatedOrNaive: 'naive'
};
var agent = makeBanditAgent(params, bandit, 'beliefDelay');
var trajectory = simulate(startState, world, agent, 'states');
var averageUtility = listMean(map(numericBanditUtility, trajectory));
print('Arm1 is best arm and has expected utility 0.5.\n' + 
      'So ideal performance gives average score of: 0.5 \n' + 
      'The average score over 100 trials for greedy agent: '
      + averageUtility);
~~~~

The next codebox is a three-arm Bandit problem show in Figure 2. Given the agent's prior, `arm0` has the highest prior expectation. So the agent will try that before exploring other arms. We show the agent's actions and their average score over 40 trials.

<img src="/assets/img/5b-greedy-bandit-2.png" alt="diagram" style="width: 400px;"/>

>**Figure 2:** Bandit problem where `arm0` has highest prior expectation for the agent but where `arm2` is actually the best arm.

~~~~
// noisy_greedy_3_arms

// agent is same as above: bound=1, alpha=10
///fold:
var greedyBound = 1;
var alpha = 10; // noise level

var params = {
  alpha: 10,
  myopia: {on: true, bound: greedyBound},
  boundVOI: {on: false, bound: 0},
  noDelays: false,
  discount: 0,
  sophisticatedOrNaive: 'naive'
};
///

var trueArmToPrizeERP = {0: categoricalERP([0.1, 0.9], [3, 0]),
	                     1: categoricalERP([0.5, 0.5], [1, 0]),
		                 2: categoricalERP([0.5, 0.5], [2, 0])};

var numberOfTrials = 40;

var bandit = makeBandit({
  numberOfArms: 3,
  armToPrizeERP: trueArmToPrizeERP,
  numberOfTrials: numberOfTrials,
  numericalPrizes: true
});
var world = bandit.world;
var startState = bandit.startState;

var agentPrior = Enumerate(function(){
  var prob3 = uniformDraw([0.1, 0.5, 0.9]);
  var prob1 = uniformDraw([0.1, 0.5, 0.9]);
  var prob2 = uniformDraw([0.1, 0.5, 0.9]);
  var armToPrizeERP = {0: categoricalERP([prob3, 1 - prob3], [3, 0]),
	                   1: categoricalERP([prob1, 1 - prob1], [1, 0]),
	                   2: categoricalERP([prob2, 1 - prob2], [2, 0])};
  return makeBanditStartState(numberOfTrials, armToPrizeERP);
});

var params = update(params, {priorBelief: agentPrior});
var agent = makeBanditAgent(params, bandit, 'beliefDelay');
var trajectory = simulate(startState, world, agent, 'stateAction');

print("Agent's first 20 actions (during exploration phase): \n" + 
      map(second,trajectory.slice(0,20)));

var averageUtility = listMean(map(numericBanditUtility, map(first,trajectory)));
print('Arm2 is best arm and has expected utility 1.\n' + 
      'So ideal performance gives average score of: 1 \n' + 
      'The average score over 40 trials for greedy agent: '
      + averageUtility);
~~~~


-------

## Myopic Exploration: the basic idea
In Chapter III.3, we noted that solving a finite-horizon Multi-arm bandit problem is intractable in the number of arms and trials. So bounded agents will use a sub-optimal but tractable algorithm for this problem. In this chapter we describe and implement a widely-studied approach to Bandits (and POMDPs generally) that is sub-optimal but which can be very effective in practice. We refer to the approach as *Myopic Exploration*, because it is "myopic" or "greedy" with respect to exploration. The idea is that the agent at time $$t_0$$ assumes he can only *explore* (i.e. update beliefs from observations) up to some cutoff point $$C_m$$ steps into the future. After that point he just *exploits* (i.e. he gain rewards but doesn't update from the rewards he observes). In reality, the agent will continue to update beliefs after time $$t_0+C_m$$. The Myopic agent, like the Naive hyperbolic discounter, has an incorrect model of his future self. We call an agent that uses Myopic Exploration a "Myopic Agent". This will be precisely defined below. 

Myopic Exploration is an efficient way to solve Bandit problems, yielding an optimal solution in the two-arm case refp:frazier2008knowledge, and also provides a good fit to human performance in Bandit problems refp:zhang2013forgetful. In what follows, we describe Myopic Exploration in more detail, explain how to incorporate it into out POMDP agent model, and then exhibit its performance on Bandit problems.

### Myopic Exploration: applications and limitations
As noted above, Myopic Exploration has been studied in Machine Learning refp:gonzalez2015glasses and Operations Research refp:ryzhov2012knowledge as part of algorithms for generalized Bandit problems. In most cases, the cutoff point $$C_m$$ after which the agent assumes himself to exploit is set to $$C_m=1$$. This results in a scalable, analytically tractable optimization problem: pull the arm that maximizes the expected value of future exploitation given you pulled that arm. This "future exploitation" means that you pick the arm that is best in expectation for the rest of time.

We've presented Bandit problems with a finite number of arms, and with discrete rewards that are uncorrelated across arms. Myopic Exploration works well in this setting but also works for generalized Bandit Problems: e.g. when rewards are correlated, when rewards are continuous, and in the "Bayesian Optimization" setting where instead of a fixed number of arms the goal is to optimize high-dimensional real-valued function refp:ryzhov2012knowledge. 

Myopic Exploration does not work well for POMDPs in general. Suppose you are looking for a good restaurant in a foreign city. A good strategy is to walk to a busy street and then find the busiest restaurant. If reaching the busy street takes longer than the myopic cutoff $$C_m$$, then a Myopic agent won't see value in this plan. We present a concrete example of this problem below ("Restaurant Search"). This example highlights a way in which Bandit problems are an especially simple POMDP. In a Bandit problem, every aspect of the unknown latent state can be queried at any timestep (by pulling the appropriate arm). So even the Myopic Agent with $$C_m=1$$ is sensitive to the information value of every possible observation that the POMDP can yield.

TODO: maybe move paragraph to footnote. 
The Myopic agent incorrectly models his future self, by assuming it ceases to update after cutoff point $$C_m$$. This incorrect "self-modeling" is also a property of model-free RL agents. For example, a Q-learner's estimation of expected utilities for states ignores the fact that the Q-learner will randomly explore with some probability. SARSA, on the other hand, does take its random exploration into account when computing this estimate. But it doesn't model the way in which its future exploration behavior will make certain actions useful in the present (as in the example of finding a restaurant in a foreign city).

### Myopic Exploration: formal model
Myopic Exploration only makes sense in the context of an agent that is capable of learning from observations (i.e. in the POMDP rather than MDP setting). So our goal is to generalize our agent model for solving POMDPs to a Myopic Exploration with $$C_m \in [1,\infty]$$.

**Exercise:** Before reading on, modify the equations defining the [POMDP agent](/chapters/3c-pomdp) in order to generalize the agent model to include Myopic Exploration. The optimal POMDP agent will be the special case when $$C_m=\infty$$.

------------

To extend the POMDP agent to the Myopic agent, we use the idea of *delays* from the previous chapter. These delays are not used to evaluate future rewards (as any discounting agent would use them). They are used to determine how future actions are simulated. If the future action occurs when delay $$d$$ exceeds cutoff point $$C_m$$, then the simulated future self does not do a belief update before taking the action. (This makes the Myopic agent analogous to the Naive agent: both simulate the future action by projecting the wrong delay value onto their future self). 

We retain the notation from the definition of the POMDP agent and skip directly to the equation for the expected utility of a state, which we modify for the Myopic agent with cutoff point $$C_m \in [1,\infty]$$:

$$
EU_{b}[s,a,d] = U(s,a) + \mathbb{E}_{s',o,a'}(EU_{b'}[s',a'_{b'},d+1])
$$

where:

- $$s' \sim T(s,a)$$ and $$o \sim O(s',a)$$

- $$a'_{b'}$$ is the softmax action the agent takes given new belief $$b'$$

- the new belief state $$b'$$ is defined as:

$$
b'(s') \propto I_{C_m}(s',a,o,d)\sum_{s \in S}{T(s,a,s')b(s)}
$$

<!-- problem with < sign in latex math-->
where $$I_{C_m}(s',a,o,d) = O(s',a,o)$$ if $$d$$ < $$C_m$$ and $$I_{C_m}(s',a,o,d) = 1$$ otherwise.

The key part is the definition of $$b'$$. The Myopic agent assumes his future self updates only on his last action $$a$$ and not on observation $$o$$. So the future self will know about state changes that follow a priori from his actions. (In a deterministic Gridworld, the future self would know his new location and that the time remaining had been counted down).

The implementation of the Myopic agent in WebPPL is a direct translation of the definition provided above.

>**Exercise:** Modify the code for the POMDP agent to represent a Myopic agent. See this <a href="/chapters/3c-pomdp.html#pomdpCode">codebox</a> or this library [script](https://github.com/agentmodels/webppl-gridworld/src/beliefAgent.wppl). <!-- TODO fix link -->


### Myopic Exploration for Bandits

The Myopic agent performs well on a variety of Bandit problems. The following codeboxes compare the Myopic agent to the Optimal POMDP agent on binary, two-arm Bandits (see the specific example in Figure 3). TODO: add statement about equivalent performance. 

<img src="/assets/img/5b-myopic-bandit.png" alt="diagram" style="width: 600px;"/>

>**Figure 3**: Bandit problem. The agent's prior includes two hypotheses for the rewards of each arm, with the prior probability of each labeled to the left and right of the boxes. The priors on each arm are independent and so there are four hypotheses overall. Boxes with actual rewards have a bold border. 
<br>

~~~~
// myopia_bandit_performance

// Helper functions for Bandits:
///fold:

// HELPERS FOR CONSTRUCTING AGENT

var baseParams = {
  alpha: 1000,
  noDelays: false,
  sophisticatedOrNaive: 'naive',
  boundVOI: {on: true, bound: 1},
  myopia: {on: false, bound: 0},
  discount: 0
};

var getParams = function(agentPrior){
  var params = update(baseParams, {priorBelief: agentPrior});
  return update(params);
};

var getAgentPrior = function(numberOfTrials, priorArm0, priorArm1){
  return Enumerate(function(){
    var armToPrizeERP = {0: priorArm0(), 1: priorArm1()};
    return makeBanditStartState(numberOfTrials, armToPrizeERP);
  });
};

// HELPERS FOR CONSTRUCTING WORLD

// Possible distributions for arms
var probably0ERP = categoricalERP([0.6, 0.4], [0, 1]);
var probably1ERP = categoricalERP([0.4, 0.6], [0, 1]);

// Construct Bandit POMDP
var getBandit = function(numberOfTrials){
  return makeBandit({
    numberOfArms: 2,
	armToPrizeERP: {0: probably0ERP, 1: probably1ERP},
	numberOfTrials: numberOfTrials,
	numericalPrizes: true
  });
};

// Get score for a single episode of bandits
var score = function(out){
  return listMean(map(numericBanditUtility, out));
};
///

// Agent prior on arm rewards

// Possible distributions for arms
var probably0ERP = categoricalERP([0.6, 0.4], [0, 1]);
var probably1ERP = categoricalERP([0.4, 0.6], [0, 1]);

// True latentState:
// arm0 is probably0ERP, arm1 is probably1ERP (and so is better)

// Agent prior on arms: arm1 (better arm) has higher EV
var priorArm0 = function(){
  return categorical([0.5, 0.5], [probably1ERP, probably0ERP]);
};
var priorArm1 = function(){
  return categorical([0.6, 0.4], [probably1ERP, probably0ERP]);
};


var runAgent = function(numberOfTrials, optimal){
  // Construct world and agents
  var bandit = getBandit(numberOfTrials);
  var world = bandit.world;
  var startState = bandit.startState;
  var prior = getAgentPrior(numberOfTrials, priorArm0, priorArm1);
  var agentParams = getParams(prior);

  var agent = optimal ? makeBanditAgent(agentParams, bandit, 'belief') :
       makeBanditAgent(agentParams, bandit, 'beliefDelay');

  return score(simulate(startState, world, agent, 'states')); 
};

// Run each agent 10 times and take average of scores
var means = map( function(optimal){
  var scores = repeat(10, function(){return runAgent(5,optimal);});
  var st = optimal ? 'Optimal: ' : 'Myopic: ';
  print(st + 'Mean scores on 10 repeats of 5-trial bandits\n' + scores);
  return listMean(scores);
  }, [true,false]);
  
print('Overall means for [Optimal,Myopic]: ' + means);
~~~~

>**Exercise**: The above codebox shows that performance for the two agents is similar. Try varying the priors and the `armToPrizeERP` and verify that performance remains similar. How would you provide stronger empirical evidence that the two algorithms are equivalent for this problem?

The following codebox computes the runtime for Myopic and Optimal agents as a function of the number of Bandit trials. We see that the Myopic agent has better scaling even on a small number of trials. Note that neither agent has been optimized for Bandit problems.

>**Exercise:** Think of ways to optimize the Myopic agent with $$C_m=1$$ for binary Bandit problems.

~~~~
// myopia_bandit_scaling
// Similar helper functions as above codebox
///fold:

// HELPERS FOR CONSTRUCTING AGENT

var baseParams = {
  alpha: 1000,
  noDelays: false,
  sophisticatedOrNaive: 'naive',
  myopia: {on: false, bound: 0},
  boundVOI: {on: true, bound: 1},
  discount: 0
};

var getParams = function(agentPrior){
  var params = update(baseParams, {priorBelief: agentPrior});
  return update(params);
};

var getAgentPrior = function(numberOfTrials, priorArm0, priorArm1){
  return Enumerate(function(){
    var armToPrizeERP = {0: priorArm0(), 1: priorArm1()};
    return makeBanditStartState(numberOfTrials, armToPrizeERP);
  });
};

// HELPERS FOR CONSTRUCTING WORLD

// Possible distributions for arms
var probably1ERP = categoricalERP([0.4, 0.6], [0, 1]);
var probably0ERP = categoricalERP([0.6, 0.4], [0, 1]);


// Construct Bandit POMDP
var getBandit = function(numberOfTrials){
  return makeBandit({
    numberOfArms: 2,
	armToPrizeERP: {0: probably0ERP, 1: probably1ERP},
	numberOfTrials: numberOfTrials,
	numericalPrizes: true
  });
};

// Get score for a single episode of bandits
var score = function(out){
  return listMean(map(numericBanditUtility, out));
};


// Agent prior on arm rewards

// Possible distributions for arms
var probably0ERP = categoricalERP([0.6, 0.4], [0, 1]);
var probably1ERP = categoricalERP([0.4, 0.6], [0, 1]);

// True latentState:
// arm0 is probably0ERP, arm1 is probably1ERP (and so is better)

// Agent prior on arms: arm1 (better arm) has higher EV
var priorArm0 = function(){
  return categorical([0.5, 0.5], [probably1ERP, probably0ERP]);
};
var priorArm1 = function(){
  return categorical([0.6, 0.4], [probably1ERP, probably0ERP]);
};


var runAgents = function(numberOfTrials){
  // Construct world and agents
  var bandit = getBandit(numberOfTrials);
  var world = bandit.world;
  var startState = bandit.startState;
  
  var agentPrior = getAgentPrior(numberOfTrials, priorArm0, priorArm1);
  var agentParams = getParams(agentPrior);

  var optimalAgent = makeBanditAgent(agentParams, bandit, 'belief');
  var myopicAgent = makeBanditAgent(agentParams, bandit, 'beliefDelay');

  // Get average score across totalTime for both agents
  var runOptimal = function(){
    return score(simulate(startState, world, optimalAgent, 'states')); 
  };
  
  var runMyopic = function(){
    return score(simulate(startState, world, myopicAgent, 'states'));
  };

  var optimalDatum = {numberOfTrials: numberOfTrials,
                      runtime: timeit(runOptimal).runtimeInMilliseconds*0.001,
					  agentType: 'optimal'
  };

  var myopicDatum = {numberOfTrials: numberOfTrials,
                     runtime: timeit(runMyopic).runtimeInMilliseconds*0.001,
					 agentType: 'myopic'
  };

  return [optimalDatum, myopicDatum];
};
///

// Compute runtime as # Bandit trials increases
var totalTimeValues = range(9).slice(2);

print('Runtime in s for [Optimal,Myopic] agents:');

var runtimeValues = _.flatten(map(runAgents, totalTimeValues));

viz.line(runtimeValues, {groupBy: 'agentType'});
~~~~


### Myopic Exploration for the Restaurant Search Problem
The limitations of Myopic exploration are straightforward. The Myopic agent assumes they will not update beliefs after the bound at $$C_m$$. As a result, they won't make plans that involve learning something after the bound.

We illustrate this limitation with a new problem:

>**Restaurant Search:** You are looking for a good restaurant in a foreign city without the aid of a smartphone. You know the quality of some restaurants already and you are uncertain about the others. If you walk right up to a restaurant, you can tell its quality by seeing how busy it is inside. You care about the quality of the restaurant (a scalar) and about minimizing the time spent walking.

How does the Myopic agent fail? Suppose that a few blocks from agent is a great restaurant next to a bad restaurant and the agent doesn't know which is which. If the agent checked inside each restaurant, they would pick out the great one. But if they are Myopic, they assume they'd be unable to tell between them.

The codebox below depicts a toy version of this problem in Gridworld. The restaurants vary in quality between 0 and 5. The agent knows the quality of Restaurant A and is unsure about the other restaurants. One of Restaurants D and E is great and the other is bad. The Optimal POMDP agent will go right up to each restaurant and find out which is great. The Myopic agent, with low enough bound $$C_m$$, will either go to the known good restaurant A or investigate one of restaurants that is closer than D and E.

TODO: Extend the x-axis to make D and E further away. Consider how to make the myopic agent faster in this context. (Would also be nice to illustrate observations or the agent's belief state somehow).

TODO: gridworld draw should take pomdp trajectories. they should also take POMDP as "world". 

~~~~
// optimal_agent_restaurant_search

var pomdp = makeRestaurantSearchPOMDP();
var world = pomdp.world;
var makeUtility = pomdp.makeUtility;
var startState = pomdp.startState;

var agentPrior = Enumerate(function(){
  var rewardD = uniformDraw([0,5]); // D is bad or great (E is opposite)
  var latentState = {A: 3,
		             B: uniformDraw(range(6)),
		             C: uniformDraw(range(6)),
		             D: rewardD,
		             E: 5 - rewardD};
  return buildState(pomdp.startState.manifestState, latentState);
});

// Construct optimal agent
var params = {
  utility: makeUtility(-0.01), // timeCost is -.01
  alpha: 1000,
  priorBelief: agentPrior
};

var agent = makePOMDPAgent(params, world);
var trajectory = simulate(pomdp.startState, world, agent, 'states');
var manifestStates = map(function(state){return state.manifestState;},
                         trajectory);
print('Quality of restaurants: \n'+JSON.stringify(pomdp.startState.latentState));
GridWorld.draw(pomdp.mdp, {trajectory: manifestStates})
~~~~

>**Exercise:** The codebox below shows the behavior the Myopic agent. Try different values for the `myopiaBound` parameter. For values in $$[1,2,3]$$, explain the behavior of the Myopic agent. 

~~~~
// myopic_agent_restaurant_search

// Construct world and agent prior as above
///fold: 
var pomdp = makeRestaurantSearchPOMDP();
var world = pomdp.world;
var makeUtility = pomdp.makeUtility;

var agentPrior = Enumerate(function(){
  var rewardD = uniformDraw([0,5]); // D is bad or great (E is opposite)
  var latentState = {A: 3,
		             B: uniformDraw(range(6)),
		             C: uniformDraw(range(6)),
		             D: rewardD,
		             E: 5 - rewardD};
  return buildState(pomdp.startState.manifestState, latentState);
  });
///

var myopiaBound = 1;

var params = {
  utility: makeUtility(-0.01),
  alpha: 1000,
  priorBelief: agentPrior,
  noDelays: false,
  discount: 0,
  sophisticatedOrNaive: 'naive',
  myopia: {on: false, bound: 0},
  boundVOI: {on: true, bound: myopiaBound},
};

var agent = makePOMDPAgent(params, world);
var trajectory = simulate(pomdp.startState, world, agent, 'states');
var manifestStates = map(function(state){return state.manifestState},
                         trajectory);

print('Rewards for each restaurant: ' + JSON.stringify(pomdp.startState.latentState));
print('Myopia bound: ' + myopiaBound);
GridWorld.draw(pomdp.mdp, {trajectory: manifestStates});
~~~~

