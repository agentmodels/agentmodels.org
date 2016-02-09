---
layout: chapter
title: Reasoning about agents
description: Overview of inverse planning / IRL. WebPPL examples of inferring utilities and beliefs from choices (online and batch).
is_section: true
---

## TODOs:
* what's up with the visualisation?
* fill in words
* visualisation of trajectories doesn't show final action: how to fix this?

# this is the real bit

Explain the idea of IRL.

## Conditioning on a single action

We're in donutWorld. This is what it looks like. There are stores.

~~~
var params = makeDonutInfer(true, {'donutSouth': 1, 'donutNorth': 1, 'veg': 1,
                                   'noodle': 1, 'timeCost': 1}, 100, 0);
GridWorld.draw(params);
~~~

This is how you infer based on a single action.

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

viz.print(inferSingleAction([2,1], ["l"], 7, simpleUtilPrior))
// viz.print(inferSingleAction([2,1], ["u"], 7, simpleUtilPrior))
~~~

we figured out that donuts are better than veg/noodles. can we figure out whether veg is better than noodles?

~~~

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

viz.print(inferSingleAction([2,1], ["l"], 7, complexUtilPrior))
// viz.print(inferSingleAction([2,1], ["u"], 7, complexUtilPrior))
~~~

nope. the actions don't tell us because the agent would do the same thing in either case.

## Conditioning on a trajectory

Do it like this. Am conditioning on actions rather than trajectories because that's what gives the relevant information, because Markov.

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

var params = makeDonutInfer(true, {'donutSouth': 1, 'donutNorth': 1, 'veg': 1, 'noodle': 1, 'timeCost': 1}, 100, 0);
var noodleTraj = [[[2,1], 'u'], [[2,2], 'u'], [[2,3], 'r']];
GridWorld.draw(params, {trajectory: noodleTraj});
// viz.print(inferTrajUtil(noodleTraj, 7, complexUtilPrior));
~~~

Note that utility functions where noodles have the same utility as veggies are equally likely as those where noodles have strictly higher utility. This is because if there were a tie, the agent would go to the closest shop to minimise the time cost, which is the noodle place.

## inferring time cost from a trajectory

this is how you do that

~~~
var inferTrajUtilTimeCost = function(trajectory, perceivedTotalTime, utilityPrior) {
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

	    var actionERPs = mdpSimulate(startState, trajectory.length,
		                             perceivedTotalTime, newParams,
								 	 outputParams).stateActionERPs;
        var erpActionPairs = zip(actionERPs, actionArray);

	    map(function(pair){factor(pair[0].score([], pair[1]))},
			erpActionPairs);

	    return {donutUtil: newUtilityTable['donutSouth'],
		        vegUtil: newUtilityTable['veg'],
				noodleUtil: newUtilityTable['noodle']
				timeCost: newUtilityTable['timeCost']};
    });
};

var superComplexUtilPrior = function() {
    var donutUtil = uniformDraw([1,2,3]);
    var vegUtil = uniformDraw([1,2,3]);
    var noodleUtil = uniformDraw([1,2,3]);
    var timeCost = uniformDraw([-0.1, -1]);
    return {'donutSouth': donutUtil,
    	    'donutNorth': donutUtil,
	    	'veg': vegUtil,
	        'noodle': noodleUtil,
    	    'timeCost': timeCost};
};

var dsTraj = [[[2,1], 'l'], [[1,1], 'l']];
GridWorld.draw(params, {trajectory: dsTraj});
// viz.print(inferTrajUtilTimeCost(dsTraj, 7, superComplexUtilPrior));
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

var dnTraj = [[[2,1], 'u'], [[2,2], 'u'], [[2,3], 'u'], [[2,4], 'l']];
GridWorld.draw(params, {trajectory: dnTraj})

var crazyTrajs = [dsTraj, noodleTraj];

var maybeTipsyPrior = function() {
    return categorical([0.1, 0.9], [10, 100]);
};

// viz.print(inferTrajsUtilAlpha(dnTrajs, [7,7], complexUtilPrior, maybeTipsyPrior))
// viz.print(inferTrajsUtilAlpha(crazyTrajs, [7,7], complexUtilPrior, maybeTipsyPrior))
~~~

now you have learned.
