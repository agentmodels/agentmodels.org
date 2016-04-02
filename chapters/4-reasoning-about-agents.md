---
layout: chapter
title: Reasoning about agents
description: Overview of inverse planning / IRL. WebPPL examples of inferring utilities and beliefs from choices (online and batch).
is_section: true
---


## Introduction
The previous chapters have shown how to compute optimal actions for agents in MDPs and POMDPs. In many practical applications, this is all we want to compute. For example, if we are controlling a robot, we would want the robot to act optimally given the utility function we have designed for it. If we want to come up with an optimal gambling strategy, we might use a POMDP agent model like that used for bandits in the [previous chapter](/chapters/3c-pomdp).

In other settings, however, our goal is to learn or reason about an agent based on their behavior. For example, in social science or psychology, researchers would like to learn about people's preferences (e.g. for spending vs. saving money, for one environmental policy vs. another) and people's beliefs. The relevant *data* is usually observations of human choices, sometimes under experimental settings. In this setting, models of optimal action are *generative models* or ("forward" models) of human behavior. The human's beliefs or utilities can be inferred from their actions by *inverting* the model using an array of statistical inference techniques. For concrete examples from economics and artificial intelligence, see refp:aguirregabiria2010dynamic, refp:darden2010smoking, and refp:ermon2014learning. 

Agent models are also used as generative models in Machine Learning, under the label "Inverse Reinforcement Learning" (IRL). One motivation for learning human preferences and beliefs is to give humans helpful recommendations (e.g. for products they are likely to enjoy). A different motivation for IRL on humans is as a technique for mimicking human expert performance on a specific task (refp:abbeel2004apprenticeship).

This chapter provides an array of illustrative examples of learning about agents from their actions. We begin with a concrete example and then provide a general formalization of the inference problem. A virtue of using WebPPL is that doing inference over our existing agent models requires very little extra code. 


## Learning about an agent from their actions: motivating example

Consider the MDP version of Bob's Restaurant Choice problem. Bob is choosing between restaurants and has full knowledge of which restaurants are open (i.e. all of them) and knows the street layout. Previously, we discussed how to compute optimal behavior *given* Bob's utility function over restaurants. Now we get to observe Bob's behavior and our task is to infer his utility function:

~~~~
var world = restaurantChoiceMDP; 
var observedPath = restaurantNameToPath.donutSouth;
var observedTrajectory = locationsToManifestStates(observedPath);

GridWorld.draw(world,{trajectory: observedTrajectory});
~~~~

From Bob's actions, we infer that he probably prefers the Donut Store to the other restaurants. An alternative explanation is that Bob cares most about saving time. He might prefer the Vegetarian Cafe (all things being equal) but his preference is not strong enough to spend extra time getting there.

In this first example of inference, Bob's preference for saving time are taken as given (with only a weak preference) and we infer (given the actions shown above) Bob's preference for the different restaurants. We model Bob using the MDP agent model from [Chapter III.1](/chapters/3a-mdp.html). We place a uniform prior over three possible utility functions for Bob: one favoring the Donut Store, one favoring Vegetarian Cafe and one favoring Noodle Shop. We use `Enumerate` to compute a Bayesian posterior over these utility functions, given Bob's observed behavior. Since the world is practically deterministic (with softmax parameter $$\alpha$$ set high), we just compare Bob's predicted states under each utility function to the states actually observed. To predict Bob's states for each utility function, we use the function `simulate` from [Chapter III.1](/chapters/3a-mdp.html). 

~~~~
var world = restaurantChoiceMDP;
var feature = world.feature;

var observedLocs = restaurantNameToPath.donutSouth;
var startState = {loc: [3,1],
		          timeLeft: 15,
				  terminateAfterAction: false};

var utilityTablePrior = function(){
  var baseUtilityTable = {
    'Donut S': 1,
    'Donut N': 1,
    Veg: 1,
    Noodle: 1,
    timeCost: -0.05
  };
  return uniformDraw( 
    [{table: update(baseUtilityTable, {'Donut N':2, 'Donut S':2}),
      favourite: 'donut'},
     {table: update(baseUtilityTable, {Veg:2}),
      favourite: 'veg'},
     {table: update(baseUtilityTable, {Noodle:2}),
      favourite: 'noodle'}]
  );
};

var posterior = Enumerate( function(){
  var utilityTableAndFavourite = utilityTablePrior();
  var utilityTable = utilityTableAndFavourite.table;
  var favourite = utilityTableAndFavourite.favourite;
  
  var utility = mdpTableToUtilityFunction(utilityTable, feature);
  var params = {utility: utility,
		        alpha: 100};
  var agent  = makeMDPAgent(params, world);
  
  var predictedStates = simulateMDP(startState, world, agent, 'states');
  var predictedLocs = _.map(predictedStates, 'loc');
  condition(_.isEqual(observedLocs, predictedLocs));
  return {favourite: favourite};
});

viz.vegaPrint(posterior);
~~~~

## Learning about an agent from their actions: formalization

We will now formalize the kind of inference in the previous example. We begin by considering inference over the utilities and softmax noise parameter for an MDP agent. Later on we'll generalize to POMDP agents and to other agents.

Following [Chapter III.1](/chapters/3a-mdp.html) the MDP agent is defined by a utility function $$U$$ and softmax parameter $$\alpha$$. In order to do inference, we need to know the agent's starting state $$s_0$$ (which might include both their *location* and their *time horizon* $$N$$). The data we condition on is a sequence of state-action pairs: 

$$
(s_0, a_0), (s_1, a_1), \ldots, (s_n, a_n)
$$

The index for the final timestep is less than or equal to the time horzion:  $$n \leq N$$. We abbreviate this sequence as $$(s,a)_{0:n}$$. The joint posterior on the agent's utilities and noise given the observed state-action sequence is:

$$
P(U,\alpha | (s,a)_{0:n}) \propto P( {(s,a)}_{0:n} | U, \alpha) P(U, \alpha)
$$

where the likelihood function $$P( {(s,a)}_{0:n} \vert U, \alpha )$$ is the MDP agent model (for simplicity we omit information about the starting state). Due to the Markov Assumption for MDPs, the probability of an agent's action in a state is independent of the agent's previous or later actions (given $$U$$ and $$\alpha$$). This allows us to rewrite the posterior as **Equation (1)**:

$$
P(U,\alpha | (s,a)_{0:n}) \propto P(U, \alpha) \prod_{i=0}^n P( a_i | s_i, U, \alpha)
$$


The term $$P( a_i \vert s_i, U, \alpha)$$ can be rewritten as the softmax choice function (which corresponds to the function `act` in our MDP agent models). This equation holds for the case where we observe a sequence of actions from timestep 0 to $$n \leq N$$ (with no gaps). This tutorial focuses mostly on this case. It is trivial to extend the equation to observing multiple independently drawn such sequences (as we show below). However, if there are gaps in the sequence or if we observe only the agent's states (not the actions), then we need to marginalize over actions that were unobserved.


## Examples of learning about agents in MDPs

### Example: Inference from part of a sequence of actions

The expression for the joint posterior (above) shows that it is straightforward to do inference on a part of an agent's action sequence. For example, if we know an agent had a time horizon of 10, we can do inference from only the agent's first few actions.

For this example, we condition on the agent making a single step from $$[3,1]$$ to $$[2,1]$$ by moving left. For an agent with low noise, this provides almost as much evidence as the agent going all the way to Donut South. 

[TODO Codebox showing trajectory with only a single step]

~~~~
var world = restaurantChoiceMDP;
var singleStepTrajectory = [{loc: [3,1],
                             timeLeft: 10,
							 terminateAfterAction: false},
						    {loc: [2,1],
							 timeLeft: 9,
							 terminateAfterAction: false}];

GridWorld.draw(world,{trajectory: singleStepTrajectory});
~~~~

Our approach to inference is slightly different than in the example at the start of this chapter. The approach is a direct translation of the expression for the posterior in Equation (1) above. For each observed state-action pair, we compute the likelihood of the agent (with given $$U$$) choosing that action in the state. (In contrast, the simple approach above will become intractable for long, noisy action sequences -- as it will need to loop over all possible sequences). 

~~~~
var world = restaurantChoiceMDP;
var feature = world.feature;

var utilityTablePrior = function(){
  var baseUtilityTable = {
    'Donut S': 1,
    'Donut N': 1,
    Veg: 1,
    Noodle: 1,
    timeCost: -0.05
  };
  return uniformDraw( 
    [{table: update(baseUtilityTable, {'Donut N':2, 'Donut S':2}),
      favourite: 'donut'},
     {table: update(baseUtilityTable, {Veg:2}),
      favourite: 'veg'},
     {table: update(baseUtilityTable, {Noodle:2}),
      favourite: 'noodle'}]
  );
};
var alpha = 100;
var observedStateAction = [[{loc: [3,1],
			                 timeLeft: 10,
							 terminateAfterAction: false}, 'l']];

var posterior = Enumerate( function(){
  var utilityTableAndFavourite = utilityTablePrior();
  var utilityTable = utilityTableAndFavourite.table;
  var utility = mdpTableToUtilityFunction(utilityTable, feature);
  var favourite = utilityTableAndFavourite.favourite;

  var params = {utility: utility,
		        alpha: alpha};
  var agent  = makeMDPAgent(params, world);
  var act = agent.act;
  // For each observed state-action pair, compute likekihood of action
  map( function(stateAction){
    factor( act(stateAction[0]).score( [], stateAction[1]) );
  }, observedStateAction );

  return {favourite: favourite};
});

viz.vegaPrint(posterior);
~~~~

Note that utility functions where Vegetarian Cafe or Noodle Shop are most preferred have almost the same posterior probability. Since they had the same prior, this means that we haven't received evidence about which the agent prefers. Moreover, assuming the agent's `timeCost` really is negligible (and the agent always has enough total timesteps), then no matter where the agent is placed on the grid, they will choose Donut North or South. So we'd never get any information about whether they prefer the Vegetarian Cafe or Noodle Shop!

Actually, this is not quite right. If we wait long enough, the agent's softmax noise would eventually reveal information about which was preferred. However, the general point remains that we won't be able to *efficiently* learn the agent's preferences by repeatedly watching them choose from a random start point. If there is no softmax noise, then we can make the strong claim that even in the limit, the agent's preferences are not *identified* by draws from this space of scenarios.

This issue of *unidentifiability* is common when inferring an agent's beliefs or utilities from realistic datasets. First, an agent (even with some softmax noise) may reliably avoid inferior states (as in the present example); and so their actions may communicate little about the relative utilities *among* the inferior states. Second, richer models of agents (e.g. those with softmax noise and inaccurate beliefs) allow for more possible explanations of the same behavior. One solution to unidentifiability for IRL is *active learning*: see refp:amin2016towards. 


### Example: Inferring The Cost of Time and Softmax Noise
The previous examples assumed that the agent's `timeCost` (the negative utility of each timestep before the agent reaches a restaurant) and the softmax $$\alpha$$ were known. We can modify the above example to include them in inference.

~~~~
var world = restaurantChoiceMDP;
var feature = world.feature;

var utilityTablePrior = function(){
  var foodValues = [0,1,2,3];
  var timeCostValues = [-0.1, -0.3, -0.6, -1];
  var donut = uniformDraw(foodValues);

  return {'Donut N': donut,
          'Donut S': donut,
          Veg: uniformDraw(foodValues),
          Noodle: uniformDraw(foodValues),
          timeCost: uniformDraw(timeCostValues)};
};
var alphaPrior = function(){return uniformDraw([.1,1,10,100]);};

var posterior = function(observedStateActionSequence){
  return Enumerate( function() {
    var utilityTable = utilityTablePrior();
    var alpha = alphaPrior();
	var logAlpha = Math.log10(alpha);
	var timeCost = utilityTable.timeCost;
    var params = {utility: mdpTableToUtilityFunction(utilityTable, feature),
		  alpha: alpha};
    var agent = makeMDPAgent(params, world);
    var act = agent.act;

    var donutBest = utilityTable['Donut N'] >= utilityTable['Veg']
	  && utilityTable['Donut N'] >= utilityTable['Noodle'];

    // For each observed state-action pair, compute likekihood of action
    map( function(stateAction){
      factor( act(stateAction[0]).score( [], stateAction[1]) );
    }, observedStateActionSequence );

    return {donutBest: donutBest, logAlpha: logAlpha,
		    timeCost: timeCost};
  });
};


var observedStateActionSequence = locationsToStateActions(restaurantNameToPath.donutSouth);

// these ERPs do not print for some reason
// TODO: fix this
// viz.vegaPrint(posterior(observedStateActionSequence.slice(0,1)));
// viz.vegaPrint(posterior(observedStateActionSequence.slice(0,2)));
viz.vegaPrint(posterior(observedStateActionSequence.slice(0,3)));
~~~~

The posterior shows that taking a step towards Donut South can now be explained in terms of a high `timeCost`. If the agent has a low value for $$\alpha$$, then this step to the left is fairly likely even if the agent prefers the Noodle Store or Vegetarian Cafe. So including softmax noise in the inference makes inferences about other parameters closer to the prior. However, once we observe three steps towards Donut South, the inferences about preferences become fairly strong. 

As noted above, it is simple to extend our approach to inference to conditioning on multiple sequences of actions. Consider the two sequences below:

~~~~
var world = restaurantChoiceMDP

var observedPath1 =  restaurantNameToPath.donutSouth;
var observedTrajectory1 = locationsToManifestStates(observedPath1);
var observedPath2 =  restaurantNameToPath.donutNorth;
var observedTrajectory2 = locationsToManifestStates(observedPath2);

GridWorld.draw(world, {trajectory: observedTrajectory1});
GridWorld.draw(world, {trajectory: observedTrajectory2});
~~~~

inference happens here

~~~~
var world = restaurantChoiceMDP;
var feature = world.feature;

var utilityTablePrior = function(){
  var foodValues = [0,1,2,3];
  var timeCostValues = [-0.1, -0.3, -0.6, -1];
  var donut = uniformDraw(foodValues);

  return {'Donut N': donut,
          'Donut S': donut,
          Veg: uniformDraw(foodValues),
          Noodle: uniformDraw(foodValues),
          timeCost: uniformDraw(timeCostValues)};
};
var alphaPrior = function(){return uniformDraw([.1,1,10,100]);};

var posterior = function(observedStateActionSequence){
  return Enumerate( function() {
    var utilityTable = utilityTablePrior();
    var alpha = alphaPrior();
    var params = {utility: mdpTableToUtilityFunction(utilityTable, feature),
		          alpha: alpha};
    var agent = makeMDPAgent(params, world);
    var act = agent.act;

    var donutBest = utilityTable['Donut N'] >= utilityTable['Veg']
	  && utilityTable['Donut N'] >= utilityTable['Noodle'];

    map( function(stateAction){
      factor( act(stateAction[0]).score( [], stateAction[1]) );
    }, observedStateActionSequence );

    return {donutBest: donutBest, logalpha: Math.log10(alpha),
	        timeCost: utilityTable.timeCost};
  });
};
  
var observedSequence1 =  locationsToStateActions(restaurantNameToPath.donutSouth);
var observedSequence2 =  locationsToStateActions(restaurantNameToPath.donutNorth);
// same problem with printing the posterior as previous codebox
// posterior(observedSequence1.concat(observedSequence2));
  // TODO: alternatively: can we run *score* on an array for more efficient computation of likelihoods
  // e.g. score([],[x1,x2]), where the function computes sufficient statistics of the input 
  
~~~~



## Learning about agents in POMDPs

### Formalization
We can extend our approach to inference to deal with agents that solve POMDPs. One approach to inference is simply to generate full state-action sequences and compare them to the observed data. As we mentioned above, this approach becomes intractable in cases where noise (in transitions and actions) is high and sequences are long.

Instead, we extend the approach in Equation (1) above. The first thing to notice is that Equation (1) has to be amended for POMDPs. In an MDP, actions are independent given $$U$$, $$\alpha$$ and the state; while in a POMDP, actions are only independent if we also condition on the *belief*. So Equation (1) can only be extended to the case where we know the agent's belief at each timestep. This will be realistic in some applications and not others. This depends on whether we observe the agent's observations (as well as their states and actions). If so, we can compute the agent's belief at each timestep (up to knowledge of their prior). If not, we have to marginalize over the possible observations, making for a more complex inference computation. 

Here is the extension of Equation (1) to the POMDP case, where we assume access to the agent's observations. Our goal is to compute a posterior on the parameters of the agent. These include $$U$$ and $$\alpha$$ as before but also the agent's initial belief $$b_0$$. 

We observe a sequence of state-observation-action triples:

$$
(s_0,o_0,a_0), (s_1,o_1,a_1), \ldots, (s_n,o_n,a_n)
$$

The index for the final timestep is at most the time horzion:  $$n \leq N$$. The joint posterior on the agent's utilities and noise given the observed sequence is:

$$
P(U,\alpha, b_0 | (s,o,a)_{0:n}) \propto P( (s,o,a)_{0:n} | U, \alpha, b_0)P(U, \alpha, b_0)
$$

To produce a factorized form of this posterior analogous to Equation (1), we compute the sequence of agent beliefs. This is given by the recursive Bayesian belief update described in [Chapter III.3](/chapters/3c-pomdp):

$$
b_i = b_{i-1} \vert s_i, o_i, a_{i-1}
$$

$$
b_i(s_i) \propto 
O(s_i,a_{i-1},o_i) 
\sum_{s_i \in S} { T(s_{i-1}, a_{i-1}, s_i) b_{i-1}(s_{i-1})}
$$

The posterior can thus be written as **Equation (2)**:

$$
P(U, \alpha, b_0 | (s,o,a)_{0:n}) \propto P(U, \alpha, b_0) \prod_{i=0}^n P( a_i | s_i, b_i, U, \alpha)
$$


### Application: IRL Bandits

To learn the preferences and beliefs of a POMDP agent we translate Equation (2) into WebPPL. In later chapters, we apply this to the Restaurant Choice problem. Here we focus on the Bandit problems introduced in the [previous chapter](/chapters/3c-pomdp).

In the bandit problems we considered, there is an unknown mapping from arms to prizes (or distributions on prizes) and the agent has preferences over these prizes. The agent will try out arms to discover the mapping and then exploit the arm that seems best. In the inverse problem ("IRL bandits"), we already know the mapping from arms to prizes and we need to infer the agent's preferences over prizes and their initial belief about the mapping.

Often the agent's choices admit of multiple explanations. Recall the deterministic example in the previous chapter when (according to the agent's belief) `arm0` had the prize "chocolate" and `arm1` either had either "champagne" or "nothing". Suppose we observe the agent chosing `arm0` on the first of five trials. If we don't know the agent's utilities or beliefs, then this choice could be explained by either:

(a) the agent's preference for chocolate over champagne, or

(b) the agent's belief that `arm1` is very likely (e.g. 95%) to deterministically yield the "nothing" prize

Given this choice by the agent, we won't be able to identify which of (a) and (b) is true (since exploration becomes less valuable every trial).

The codebox below implements this example. The translation of Equation (2) is in the function `factorSequence`. This function iterates through the observed state-observation-action triples, updating the agent's belief at each timestep. Thus it interleaves conditioning on an action (via `factor`) with computing the sequence of belief functions $$b_i$$. The variable names correspond as follows:

- $$b_0$$ is `initialBelief` (an argument to `factorSequence`)

- $$s_i$$ is `state`

- $$b_i$$ is `nextBelief`

- $$a_i$$ is `observedAction`

~~~~
var agentModelsIRLBanditInfer = function(baseAgentParams, priorPrizeToUtility,
                                         priorInitialBelief, worldAndStart,
										 observedSequence){

  return Enumerate(function(){
    var prizeToUtility = sample(priorPrizeToUtility);
    var initialBelief = sample(priorInitialBelief);
    
    var agent = makeIRLBanditAgent(prizeToUtility,
	                               update(baseAgentParams,
								          {priorBelief:initialBelief}),
								   worldAndStart, 'belief');
    var agentAct = agent.act;
    var agentUpdateBelief = agent.updateBelief;
    
    var factorSequence = function(currentBelief, previousAction, timeIndex){
      if (timeIndex < observedSequence.length) { 
        var state = observedSequence[timeIndex].state;
        var observation = observedSequence[timeIndex].observation;
        var nextBelief = agentUpdateBelief(currentBelief, observation,
		                                   previousAction);
        var nextActionERP = agentAct(nextBelief);
        var observedAction = observedSequence[timeIndex].action;
        
        factor(nextActionERP.score([], observedAction));
        
        factorSequence(nextBelief, observedAction, timeIndex + 1);
      }
    };
    factorSequence(initialBelief,'noAction', 0);
    
    return {prizeToUtility: prizeToUtility, priorBelief:initialBelief};
  });
};
~~~~

- Need to generate the state-observation-action triples for this example. They should have structure {state:, observation:, action:} as indicated in the code. 

- Start with an easier example than the one mentioned in the main text. The agent decides to explore (takes arm1), gets champagne and then takes arm1 thereafter. So we know the agent must prefer champagne to chocolate. Do this just with a prior on agent's utilities and a delta on his beliefs.

~~~~
var armToPrize = {0: 'chocolate',
		          1: 'champagne'};
var worldAndStart = makeIRLBanditWorldAndStart(2, armToPrize, 5);
var observe = worldAndStart.world.observe;
var fullObserve = getFullObserve(observe);
var transition = worldAndStart.world.transition;

var makeTrajectory = function(state) {
  var observation = fullObserve(state);
  var action = 1; // agent always pulls arm 1
  var nextState = transition(state, action);
  var out = {state: state,
	         observation: observation,
	         action: action};
  if (state.manifestState.terminateAfterAction) {
    return out;
  } else {
    return cons(out, makeTrajectory(nextState));
  }
};

var observedSequence = makeTrajectory(worldAndStart.startState);

var baseParams = {
  alpha: 100
};

var agentPrior = Enumerate(function(){
  var latent = flip(0.5) ? armToPrize : update(armToPrize, {1: 'nothing'});
  return buildState(worldAndStart.startState.manifestState, latent);
});
var priorInitialBelief = deltaERP(agentPrior);

var likesChampagne = {nothing: 0,
		              champagne: 5,
					  chocolate: 3};
var likesChocolate = {nothing: 0,
		              champagne: 3,
					  chocolate: 5};

var priorPrizeToUtility = categoricalERP([0.5, 0.5], [likesChampagne,
						                              likesChocolate]);

var posterior = agentModelsIRLBanditInfer(baseParams, priorPrizeToUtility,
					                      priorInitialBelief, worldAndStart,
										  observedSequence);

var chocolateUtilityPosterior = Enumerate(function(){
  var utilityBelief = sample(posterior);
  var likesChocolate = utilityBelief.prizeToUtility.chocolate > 3;
  return {likesChocolate: likesChocolate};
});
  
viz.vegaPrint(chocolateUtilityPosterior);
~~~~

- Then do example mentioned above where we condition on the agent taking arm0 for the first action. In this example, if the agent doesn't explore first time, then they won't explore at all. So additional observations wouldn't make a difference.

~~~~
var armToPrize = {0: 'chocolate',
		          1: 'champagne'};
var worldAndStart = makeIRLBanditWorldAndStart(2, armToPrize, 5);
var observe = worldAndStart.world.observe;
var fullObserve = getFullObserve(observe);
var transition = worldAndStart.world.transition;

var makeTrajectory = function(state) {
  var observation = fullObserve(state);
  var action = 0; // agent always pulls arm 0
  var nextState = transition(state, action);
  return [{state: state,
	       observation: observation,
	       action: action}];
};

var observedSequence = makeTrajectory(worldAndStart.startState);

var baseParams = {
  alpha: 100
};

var noChampagnePrior = Enumerate(function(){
  var latent = flip(0.05) ? armToPrize : update(armToPrize, {1: 'nothing'});
  return buildState(worldAndStart.startState.manifestState, latent);
});
var informedPrior = deltaERP(worldAndStart.startState);
var priorInitialBelief = categoricalERP([0.5, 0.5], [noChampagnePrior,
						                             informedPrior]);

var likesChampagne = {nothing: 0,
		              champagne: 5,
					  chocolate: 3};
var likesChocolate = {nothing: 0,
		              champagne: 3,
					  chocolate: 5};

var priorPrizeToUtility = categoricalERP([0.5, 0.5], [likesChampagne,
						                              likesChocolate]);

var posterior = agentModelsIRLBanditInfer(baseParams, priorPrizeToUtility,
					                      priorInitialBelief, worldAndStart,
										  observedSequence);
var utilityBeliefPosterior = Enumerate(function(){
  var utilityBelief = sample(posterior);
  var chocolateUtility = utilityBelief.prizeToUtility.chocolate;
  var likesChocolate = chocolateUtility > 3;
  var isInformed = isDeltaERP(utilityBelief.priorBelief);
  return {likesChocolate: likesChocolate,
	      isInformed: isInformed};
});
viz.vegaPrint(utilityBeliefPosterior);
~~~~

- If we increase the total time and leave everything else fixed, then we'll get a stronger inference about the preference for chocolate over champagne. (Because if agent prefers champagne, then even a low prior on arm1 yielding chocolate will make exploration worth it as the total time gets long enough). Show a graph of how the preference increases with timeLeft.

~~~~
var probLikesChocolate = function(timeLeft){
  var armToPrize = {0: 'chocolate',
		            1: 'champagne'};
  var worldAndStart = makeIRLBanditWorldAndStart(2, armToPrize, timeLeft);
  var observe = worldAndStart.world.observe;
  var fullObserve = getFullObserve(observe);
  var transition = worldAndStart.world.transition;

  var makeTrajectory = function(state) {
    var observation = fullObserve(state);
    var action = 0; // agent always pulls arm 0
    var nextState = transition(state, action);
    return [{state: state,
	         observation: observation,
	         action: action}];
  };

  var observedSequence = makeTrajectory(worldAndStart.startState);

  var baseParams = {
    alpha: 100
  };

  var noChampagnePrior = Enumerate(function(){
    var latent = flip(0.2) ? armToPrize : update(armToPrize, {1: 'nothing'});
    return buildState(worldAndStart.startState.manifestState, latent);
  });
  var informedPrior = deltaERP(worldAndStart.startState);
  var priorInitialBelief = categoricalERP([0.5, 0.5], [noChampagnePrior,
						                               informedPrior]);

  var likesChampagne = {nothing: 0,
			            champagne: 5,
						chocolate: 3};
  var likesChocolate = {nothing: 0,
			            champagne: 3,
						chocolate: 5};

  var priorPrizeToUtility = categoricalERP([0.5, 0.5], [likesChampagne,
							                            likesChocolate]);

  var posterior = agentModelsIRLBanditInfer(baseParams, priorPrizeToUtility,
					                        priorInitialBelief, worldAndStart,
											observedSequence);

  var likesChocInformed = {prizeToUtility: likesChocolate,
			               priorBelief: informedPrior};
  var probLikesChocInformed = Math.exp(posterior.score([], likesChocInformed));
  var likesChocNoChampagne = {prizeToUtility: likesChocolate,
			                  priorBelief: noChampagnePrior};
  var probLikesChocNoChampagne = Math.exp(posterior.score([], likesChocNoChampagne));
  return probLikesChocInformed + probLikesChocNoChampagne;
};

var lifetimes = [5,6,7,8,9];
var probsLikesChoc = map(probLikesChocolate, lifetimes);

print('Probability of liking chocolate for lifetimes ' + lifetimes + '\n'
      + probsLikesChoc);

viz.bar(lifetimes, probsLikesChoc)
~~~~

]

We include an inference function which is based on `factorOffPolicy` in irlBandits.wppl. We specialize this to the beliefAgent and add observations to the `observedStateAction`. The function is currently in irlBandits.wppl as *agentModelsIRLBanditInfer*].


This example of inferring an agent's utilities from a bandit problem may seem contrived. However, there are more practical problems that have a similar structure. Consider a domain where $$k$$ *sources* (arms) produce a stream of content, with each piece of content having a *category* (prizes). At each timestep, a human is observed choosing a source. The human has uncertainty about the stochastic mapping from sources to categories. Our goal is to infer the human's beliefs about the sources and their preferences over categories. The sources could be blogs or feeds that tag posts/tweets using the same set of tags. Alternatively, the sources could be channels for TV shows or songs. In this kind of application, the same issue of identifiability arises. An agent may choose a source either because they know it produces content in the best categories or because they have a strong prior belief that it does.




--------------


[Table of Contents](/)
