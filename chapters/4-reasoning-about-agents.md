---
layout: chapter
title: Reasoning about agents
description: Overview of inverse planning / IRL. WebPPL examples of inferring utilities and beliefs from choices (online and batch).
is_section: true
---


## PLAN
- Overview and concrete example in gridworld (single action case). Then formalize the general problem assuming we get to condition from (state,actions). Then show example of state action inference.

inferOffPolicy: give as data series [(s,a)].

p(U / siai) = p(U) PI( p(ai / si,U) )
where p(ai / si,U) = C(ai ; si U)

- Then show an example of inference from multiple trajectories in gridworld.

- POMDP inference. Show simpel IRL bandits case, where we infer the agent's prior over one thing having a good prize. Maybe show naive or soph case. 

Previous chapters exhibited models for planning in MDPs and POMDPs. This chapter shows how we can add a few lines of code to our agent models in order to infer the agent's beliefs and utilities from their observed behavior. 

## Introduction
The previous chapters have shown how to compute optimal actions for agents in MDPs and POMDPs. In many practical applications, this is all we want to compute. For example, if we are controlling a robot, we would want the robot to act optimally given the utility function we have designed for it. If we want to come up with an optimal gambling strategy, we might use a POMDP agent model like that used for bandits in the [previous chapter](/chapters/3c-pomdp).

In other settings, however, our goal is to learn or reason about an agent based on their behavior. For example, in social science or psychology, researchers would like to learn about people's preferences (e.g. for spending vs. saving money, for one environmental policy vs. another) and people's beliefs. The relevant *data* is usually observations of human choices, sometimes under experimental settings. In this setting, models of optimal action are *generative models* or ("forward" models) of human behavior. The human's beliefs or utilities can be inferred from their actions by *inverting* the model using an array of statistical inference techniques. For concrete examples from economics and artificial intelligence, see refp:aguirregabiria2010dynamic, refp:darden2010smoking, and refp:ermon2014learning. 

Agent models are also used as generative models in Machine Learning, under the label "Inverse Reinforcement Learning" (IRL). One motivation for learning human preferences and beliefs is to give humans helpful recommendations (e.g. for products they are likely to enjoy). A different motivation for IRL on humans is as a technique for mimicking human expert performance on a specific task (refp:abbeel2004apprenticeship).

This chapter provides an array of illustrative examples of learning about agents from their actions. We begin with a concrete example and then provide a general formalization of the inference problem. A virtue of using WebPPL is that doing inference over our existing agent models requires very little extra code. 


## Inferring Utilities From Actions

Consider the MDP version of Bob's Restaurant Choice problem. Bob is choosing between restaurants and has full knowledge of which restaurants are open (i.e. all of them) and knows the street layout. Previously, we discussed how to compute optimal behavior *given* Bob's utility function over restaurants. Now we get to observe Bob's behavior and our task is to infer his utility function:

[Could be Donut Big or Small. Trajectory is from the normal startState to Donut South.]
~~~~
var world = makeDonutWorld2({big:true});

GridWorld.draw(params, 
     trajectory: [ [[2,1],'l'], [[1,1],'l'] ] });
~~~~

From Bob's actions, we infer that he probably prefers the Donut Store to the other restaurants. An alternative explanation is that Bob cares most about saving time. He might prefer the Vegetarian Cafe (all things being equal) but his preference is not strong enough to spend extra time getting there.

In this first example of inference, Bob's preference for saving time are taken as given (with only a weak preference) and we infer (given the actions shown above) Bob's preference for the different restaurants. We model Bob using the MDP agent model from [Chapter III.1](/chapters/3-agents-as-programs.html). We place a uniform prior over three possible utility functions for Bob: one favoring the Donut Store, one favoring Vegetarian Cafe and one favoring Noodle Shop. We use `Enumerate` to compute a Bayesian posterior over these utility functions, given Bob's observed behavior. Since the world is deterministic (with softmax parameter $$\alpha$$ set high), we just compare Bob's predicted states under each utility function to the states actually observed. To predict Bob's states for each utility function, we use the function `simulate` from [Chapter III.1](/chapters/3-agents-as-programs.html). 

[Need new version of MDP simulate that has similar form to beliefDelay: with timeLeft as part of state etc.


~~~~

var utilityTablePrior = function(){
  var baseUtilityTable = {
    'donutSouth': 1,
    'donutNorth': 1,
    'veg': 1,
    'noodle': 1,
    'timeCost': -0.1
  };
  return uniformDraw( 
    update(baseUtilityTable, {donutNorth:2, donutSouth:2}),
    update(baseUtilityTable, {veg:2}),
    update(baseUtilityTable, {noodle:2})
  );
};
var observedStates = [];
var world = makeDonutWorld2({big:true, start:[2,1], timeLeft:10});

var posterior  = Enumerate( function(){
  var utilityTable = utilityTablePrior();
  var agent  = makeMDPAgent(utilityTable, world);
  var predictedStates = simulateMDP(world, agent, 'states');
  condition( _.isEqual( observedStates, predictedStates ) );
  return utilityTable;
});

print(posterior)

~~~~












## Conditioning on a single action

We work in the Restaurant Choice MDP. Bob has preferences over restaurants and has a preference for getting food quickly. 

We display this gridworld, and the agent's starting state:

~~~
var params = makeDonutInfer(true, {'donutSouth': 1, 'donutNorth': 1, 'veg': 1,
                                   'noodle': 1, 'timeCost': -0.1}, 100, 0);
GridWorld.draw(params, {labels: params.labels, trajectory: [[[2,1]]]});
~~~

Different restaurants will have different utilities for the agent, with the only constraint being that the two donut shops have the same utility. There will also be a time cost to the agent that encourages it to reach a destination quickly. To start off with, we will have low softmax and transition noise.

Suppose we see a single action by the agent. How can we make inferences about the agent's utilities? We first display the agent making a single move to the left.

~~~
var params = makeDonutInfer(true, {'donutSouth': 1, 'donutNorth': 1, 'veg': 1,
                                   'noodle': 1, 'timeCost': -0.1}, 100, 0);
GridWorld.draw(params, 
    {labels: params.labels, 
     trajectory: [ [[2,1],'l'], [[1,1],'l'] ] });
~~~

Here is a function that does inference over utility functions given an observation. 

~~~
// startState is the state where the agent makes its action, observedAction is
// the action that we see, perceivedTotalTime is the lifetime that the agent
// thinks it has, and utilityPrior is a thunk that stochastically returns a
// table of utilities of  terminal states and time cost. all the inference is
// done by enumeration.

var inferSingleAction = function(startState, observedAction, perceivedTotalTime,
                                 utilityPrior) {
    return Enumerate(function(){
	    var newUtilityTable = utilityPrior();
		var newParams = makeDonutInfer(true, newUtilityTable, 100, 0);
	
	    var mdpSimOptions = {trajectoryNumRejectionSamples: 0,
		                     erpOverStatesOrActions: 'actions',
							 conditionOnStates: false};
        var actionERP = mdpSimulate(startState, 1, perceivedTotalTime, newParams,
							        mdpSimOptions).erp;

	    factor(actionERP.score([], observedAction));

	    return {donutUtil: newUtilityTable['donutSouth'],
			    vegUtil: newUtilityTable['veg'],
			    noodleUtil: newUtilityTable['noodle']};
    });
};

// now, we write our prior over utilities, and feed it to our inference
// function.

var simpleUtilPrior = function(){
    if (flip()) {
		return {'donutSouth': 2,
		        'donutNorth': 2,
				'veg': 1,
				'noodle': 1,
				'timeCost': -0.1};
    } else {
		return {'donutSouth': 1,
		        'donutNorth': 1,
				'veg': 2,
				'noodle': 2,
				'timeCost': -0.1};
    }
};

print('Inferred utility function after agent moves one step to the left');
inferSingleAction([2,1], ["l"], 7, simpleUtilPrior).MAP().val;

//viz.print(inferSingleAction([2,1], ["l"], 7, simpleUtilPrior))
// viz.print(inferSingleAction([2,1], ["u"], 7, simpleUtilPrior))
~~~

We inferred that the Donut Store is preferred to the Vegetarian Cafe and the Noodle Bar. Can we figure out whether Vegetarian Cafe is better than Noodles? No, this is not identifiable from this single observation. 

~~~
var printERP = function(x,k) {
  var erpValues = sort(x.support(), undefined, function(v){return -x.score([], v);});
  var erpValues = typeof(k)=='undefined' ? erpValues : erpValues.slice(0,k);
  map(
    function(v){
      var prob = Math.exp(x.score([], v));
      if (prob > 0.0){
        print(JSON.stringify(v) + ': ' + prob.toFixed(5));
      }
    },
    erpValues);
};


var inferSingleAction = function(startState, observedAction, perceivedTotalTime,
                                 utilityPrior) {
    return Enumerate(function(){
	    var newUtilityTable = utilityPrior();
		var newParams = makeDonutInfer(true, newUtilityTable, 100, 0);
	
	    var mdpSimOptions = {trajectoryNumRejectionSamples: 0,
		                     erpOverStatesOrActions: 'actions',
							 conditionOnStates: false};
        var actionERP = mdpSimulate(startState, 1, perceivedTotalTime, newParams,
							        mdpSimOptions).erp;

	    factor(actionERP.score([], observedAction));

	    return {
			    vegUtil: newUtilityTable['veg'],
			    noodleUtil: newUtilityTable['noodle']};
    });
};
	
var complexUtilPrior = function(){
    var donutUtil = uniformDraw([1,2,3]);
    var vegUtil = uniformDraw([1,2,3]);
    var noodleUtil = uniformDraw([1,2,3]);
    return {'donutSouth': donutUtil,
     	    'donutNorth': donutUtil,
			'veg': vegUtil,
			'noodle': noodleUtil,
			'timeCost': -0.1};
};

print('Inferred posterior on utilities for Veg and Noodle after move one step to the left');
print('NB: We cannot tell which of Veg and Noodle is preferred\n');
printERP(inferSingleAction([2,1], ["l"], 7, complexUtilPrior));
~~~~



## Conditioning on a trajectory [work in progress]

We can also condition on a trajectory (not just a single action of the agent). 

~~~
// second inference function: inferring restaurant utilities from a
// trajectory. a trajectory is an array of the form
// [[state1, action1], [state2, action2], ...]. utilityPrior is the same as in
// inferSingleAction. we will simulate the agent in the new mdp in the same
// states as were in the trajectory, and condition on the actions being the same.

var inferTrajUtil = function(trajectory, perceivedTotalTime, utilityPrior) {
    return Enumerate(function(){
		var newUtilityTable = utilityPrior();
		var newParams = makeDonutInfer(true, newUtilityTable, 100, 0);

	    var startState = trajectory[0][0];
		var stateArray = map(first, trajectory);   // returns the first element
			                                       // of everything in the
												   // trajectory, which is just
												   // the array of states visited.
	    var actionArray = map(second, trajectory); // similarly, this gets the
												   // array of actions made

        var outputParams = {trajectoryNumRejectionSamples: 0,
			                erpOverStatesOrActions: 'actions',
			                conditionOnStates: stateArray};
	    // this next function returns a list of ERPs over the next action
		// in each state in trajectory.					
	    var actionERPs = mdpSimulate(startState, trajectory.length,
		                             perceivedTotalTime, newParams,
								 	 outputParams).stateActionERPs;
        var erpActionPairs = zip(actionERPs, actionArray);

	    map(function(pair){factor(pair[0].score([], pair[1]))},
			erpActionPairs);

	    return {donutUtil: newUtilityTable['donutSouth'],
			    vegUtil: newUtilityTable['veg'],
			    noodleUtil: newUtilityTable['noodle']};
    });
};

var complexUtilPrior = function(){
    var donutUtil = uniformDraw([1,2,3]);
    var vegUtil = uniformDraw([1,2,3]);
    var noodleUtil = uniformDraw([1,2,3]);
    return {'donutSouth': donutUtil,
     	    'donutNorth': donutUtil,
			'veg': vegUtil,
			'noodle': noodleUtil,
			'timeCost': -0.1};
};

var params = makeDonutInfer(true, {'donutSouth': 1, 'donutNorth': 1, 'veg': 1,
                                   'noodle': 1, 'timeCost': -0.1}, 100, 0);
var noodleTrajectory = [[[2,1], 'u'], [[2,2], 'u'], [[2,3], 'r'], [[3,3], 'r']];
GridWorld.draw(params, {trajectory: noodleTrajectory, labels: params.labels});
viz.print(inferTrajUtil(noodleTrajectory, 7, complexUtilPrior));
~~~

Note that utility functions where Noodle has the same utility as Veg are equally likely as those where noodles have strictly higher utility. This is because if there were a tie, the agent would go to the closest shop to minimize the time cost, which is the noodle place.


<!--
## inferring time cost from a trajectory

this is how you do that. use MCMC because so many options for utility functions

~~~
var inferTrajUtilTimeCost = function(trajectory, perceivedTotalTime,
                                     utilityPrior) {
    return MCMC(function(){
		var newUtilityTable = utilityPrior();
		var newParams = makeDonutInfer(true, newUtilityTable, 100, 0);

	    var startState = trajectory[0][0];
		var stateArray = map(first, trajectory);   // returns the first element
			                                       // of everything in the
												   // trajectory, which is just
												   // the array of states visited.
	    var actionArray = map(second, trajectory); // similarly, this gets the
												   // array of actions made

        var outputParams = {trajectoryNumRejectionSamples: 0,
			                erpOverStatesOrActions: 'actions',
			                conditionOnStates: stateArray};

	    var actionERPs = mdpSimulate(startState, trajectory.length,
		                             perceivedTotalTime, newParams,
								 	 outputParams).stateActionERPs;
        var erpActionPairs = zip(actionERPs, actionArray);

	    map(function(pair){factor(pair[0].score([], pair[1]))},
			erpActionPairs);

	    return {donutUtil: newUtilityTable['donutSouth'],
		        vegUtil: newUtilityTable['veg'],
				noodleUtil: newUtilityTable['noodle'],
				timeCost: newUtilityTable['timeCost']};
    });
};

var superComplexUtilPrior = function() {
    var donutUtil = uniformDraw([1, 2, 3]);
    var vegUtil = uniformDraw([1, 2, 3]);
    var noodleUtil = uniformDraw([1, 2, 3]);
    var timeCost = uniformDraw([-1, -0.5, -0.1]);
    return {'donutSouth': donutUtil,
    	    'donutNorth': donutUtil,
	    	'veg': vegUtil,
	        'noodle': noodleUtil,
    	    'timeCost': timeCost};
};

var params = makeDonutInfer(true, {'donutSouth': 1, 'donutNorth': 1, 'veg': 1,
                                   'noodle': 1, 'timeCost': -0.1}, 100, 0);
var dsTrajectory = [[[2,1], 'l'], [[1,1], 'l']];
// GridWorld.draw(params, {trajectory: dsTrajectory, labels: params.labels});
// note: viz.print is pretty good here.
viz.print(inferTrajUtilTimeCost(dsTrajectory, 7, superComplexUtilPrior));
~~~

### inferring softmax noise from multiple trajectories

if you have different trajectories doing different things, your agent is probably drunk.

~~~
// add helpful comments maybe
var inferTrajsUtilAlpha = function(trajectories, perceivedTotalTimes,
                                   utilityPrior, alphaPrior) {
    return Enumerate(function(){
		var newUtilityTable = utilityPrior();
		var newAlpha = alphaPrior();
		var newParams = makeDonutInfer(true, newUtilityTable, newAlpha, 0);

		var zippedTrajTimes = zip(trajectories, perceivedTotalTimes);

	    map(function(trajAndTime){
			var startState = trajAndTime[0][0][0];
			var stateArray = map(first, trajAndTime[0]);
			var actionArray = map(second, trajAndTime[0]);
			var perceivedTotalTime = trajAndTime[1];
			var actualTotalTime = trajAndTime[0].length;

	        var outputParams = {trajectoryNumRejectionSamples: 0,
			                    erpOverStatesOrActions: 'both',
								conditionOnStates: stateArray};
	        var newTrajERP = mdpSimulate(startState, actualTotalTime,
				                         perceivedTotalTime, newParams,
										 outputParams).stateActionERPs;
	        var erpActionPairs = zip(actionERPs, actionArray);

	        map(function(pair){factor(pair[0].score([], pair[1]));},
				erpActionPairs);
			}
		, zippedTrajTimes);

	    return {donutUtil: newUtilityTable['donutSouth'],
    			vegUtil: newUtilityTable['veg'],
				noodleUtil: newUtilityTable['noodle'],
				alpha: newAlpha};
	});
};

var params = makeDonutInfer(true, {'donutSouth': 1, 'donutNorth': 1, 'veg': 1,
                                   'noodle': 1, 'timeCost': -0.1}, 100, 0);

var dnTrajectory = [[[2,1], 'u'], [[2,2], 'u'], [[2,3], 'u'], [[2,4], 'l'],
                    [[1,4], 'l']];
var dnTrajectorys = [dnTrajectory];

var noodleTrajectory = [[[2,1], 'u'], [[2,2], 'u'], [[2,3], 'r'], [[3,3], 'r']];
var dsTrajectory = [[[2,1], 'l'], [[1,1], 'l']];
var crazyTrajectories = [dsTrajectory, noodleTrajectory];

var maybeTipsyPrior = function() {
    return categorical([0.1, 0.9], [10, 100]);
};

GridWorld.draw(params, {trajectory: dnTrajectory, labels: params.labels})
// viz.print(inferTrajsUtilAlpha(dnTrajectories, [7,7], complexUtilPrior,
//                               maybeTipsyPrior))
// viz.print(inferTrajsUtilAlpha(crazyTrajectories, [7,7], complexUtilPrior,
//                               maybeTipsyPrior))
~~~

now you have learned.
-->

--------------

[Table of Contents](/)
