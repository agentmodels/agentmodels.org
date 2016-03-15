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

[TODO Could be Donut Big or Small. Trajectory is from the normal startState to Donut South.]

~~~~
var world = makeDonutWorld2({big:true});

GridWorld.draw(params, 
     trajectory: [ [[2,1],'l'], [[1,1],'l'] ] });
~~~~

From Bob's actions, we infer that he probably prefers the Donut Store to the other restaurants. An alternative explanation is that Bob cares most about saving time. He might prefer the Vegetarian Cafe (all things being equal) but his preference is not strong enough to spend extra time getting there.

In this first example of inference, Bob's preference for saving time are taken as given (with only a weak preference) and we infer (given the actions shown above) Bob's preference for the different restaurants. We model Bob using the MDP agent model from [Chapter III.1](/chapters/3a-mdp.html). We place a uniform prior over three possible utility functions for Bob: one favoring the Donut Store, one favoring Vegetarian Cafe and one favoring Noodle Shop. We use `Enumerate` to compute a Bayesian posterior over these utility functions, given Bob's observed behavior. Since the world is deterministic (with softmax parameter $$\alpha$$ set high), we just compare Bob's predicted states under each utility function to the states actually observed. To predict Bob's states for each utility function, we use the function `simulate` from [Chapter III.1](/chapters/3a-mdp.html). 

[TODO: Need new version of MDP simulate that has similar form to beliefDelay: with timeLeft as part of state etc.]

~~~~

var utilityTablePrior = function(){
  var baseUtilityTable = {
    'donutSouth': 1,
    'donutNorth': 1,
    'veg': 1,
    'noodle': 1,
    'timeCost': -0.05
  };
  return uniformDraw( 
    update(baseUtilityTable, {donutNorth:2, donutSouth:2}), // prefers Donut
    update(baseUtilityTable, {veg:2}), // prefers Veg
    update(baseUtilityTable, {noodle:2}) // prefers Noodle
  );
};
var observedStates = []; // TODO add observed states from above
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

## Learning about an agent from their actions: formalization

We will now formalize the kind of inference in the previous example. We begin by considering inference over the utilities and softmax noise parameter for an MDP agent. Later on we'll generalize to POMDP agents and to other agents.

Following [Chapter III.1](/chapters/3a-mdp.html) the MDP agent is defined by a utility function $$U$$ and softmax parameter $$\alpha$$. In order to do inference, we need to know the agent's starting state $$s_0$$ (which might include both their *location* and their *time horizon* $$N$$). The data we condition on is a sequence of state-action pairs: 

$$
(s_0, a_0), (s_1, a_1), \ldots, (s_n, a_n)
$$

The index for the final timestep is at most the time horzion:  $$n \leq N$$. We abbreviate this sequence as $$(s,a)_{0:n}$$. The joint posterior on the agent's utilities and noise given the observed state-action sequence is:

$$
P(U,\alpha | (s,a)_{0:n}) \propto P( (s,a)_{0:n} | U, \alpha) P(U, \alpha)
$$

The likelihood function $$P( (s,a)_{0:n} | U, \alpha )$$ is the MDP agent model, suppressing information about the starting state (and so on). Due to the Markov Assumption for MDPs, the probability of an agent's action in a state is independent of the agent's previous or later actions (given $$U$$ and $$\alpha$$). So posterior can be written as:

$$
P(U,\alpha | (s,a)_{0:n}) \propto P(U, \alpha) \prod_{i=0}^n P( a_i | s_i, U, \alpha)
$$

<!-- \verb!(1): ! -->

The term $$P( a_i | s_i, U, \alpha)$$ can be rewritten as the softmax choice function (which corresponds to the function `act` in our MDP agent models). This equation holds for the case where we observe a sequence of actions from timestep 0 to $$n \leq N$$ (with no gaps). This tutorial focuses mostly on this case. It is trivial to extend the equation to observing multiple independently drawn such sequences (as we show below). However, if there are gaps in the sequence or if we observe only the agent's states (not the actions), then we need to marginalize over actions that were unobserved.


## Examples of learning about agents in MDPs

### Example: Inference from part of a sequence of actions

The expression for the joint posterior (above) shows that it is straightforward to do inference on a part of an agent's action sequence. For example, if we know an agent had a time horizon of 10, we can do inference from only the agent's first few actions.

For this example, we condition on the agent making a single step from [3,1] to [2,1] by moving left. For an agent with low noise, this provides almost as much evidence as the agent going all the way to Donut South. 

[Codebox showing trajectory with only a single step]

Our approach to inference is slightly different than in the example at the start of this chapter. The approach is a direct translation of the expression for the posterior in Equation (1) above. For each observed state-action pair, we compute the likelihood of the agent (with given $$U$$) choosing that action in the state. (In contrast, the naive approach above will become intractable for long, noisy action sequences -- as it will need to loop over all possible sequences). 

~~~~
var utilityTablePrior = function(){
  var baseUtilityTable = {
    'donutSouth': 1,
    'donutNorth': 1,
    'veg': 1,
    'noodle': 1,
    'timeCost': -0.05
  };
  return uniformDraw( 
    update(baseUtilityTable, {donutNorth:2, donutSouth:2}), // prefers Donut
    update(baseUtilityTable, {veg:2}), // prefers Veg
    update(baseUtilityTable, {noodle:2}) // prefers Noodle
  );
  };
var alpha = 100;
var observedStateActionSequence = []; // TODO add observed
var world = makeDonutWorld2({big:true, start:[3,1], timeLeft:10});

var posterior = Enumerate( function(){
  var utilityTable = utilityTablePrior();
  var agent = makeMDPAgent(utilityTable, alpha, world);
  var act = agent.act;

  // For each observed state-action pair, compute likekihood of action
  map( function(stateAction){
    factor( act(stateAction.state).score( [], stateAction.action) );
  }, observedStateActionSequence )

  return utilityTable;
});

print(posterior)
~~~~

Note that utility functions where Vegetarian Cafe or Noodle Shop are most preferred have almost the same posterior probability. Since they had the same prior, this means that we haven't received evidence about which the agent prefers. Moreover, assuming the agent's `timeCost` really is negligible (and the agent always has enough total timesteps), then no matter where the agent is placed on the grid, they will choose Donut North or South. So we'd never get any information about whether they prefer the Vegetarian Cafe or Noodle Shop!

Actually, this is not quite right. If we wait long enough, the agent's softmax noise would eventually reveal information about which was preferred. However, the general point remains that we won't be able to *efficiently* learn the agent's preferences by repeatedly watching them choose from a random start point. If there is no softmax noise, then we can make the strong claim that even in the limit, the agent's preferences are not *identified* by draws from this space of scenarios.

This issue of *unidentifiability* is common when inferring an agent's beliefs or utilities from realistic datasets. First, an agent (even with some softmax noise) may reliably avoid inferior states (as in the present example); and so their actions may communicate little about the relative utilities *among* the inferior states. Second, richer models of agents (e.g. those with softmax noise and inaccurate beliefs) allow for more possible explanations of the same behavior. One solution to unidentifiability for IRL is *active learning* or *experimental design* (see refp:amin2016towards). 


### Example: Inference on `timeCost` and Softmax Noise
The previous examples assumed that the agent's `timeCost` (the negative utility of each timestep before the agent reaches a restaurant) and the softmax $$\alpha$$ were known. We can modify the above example to include them in inference.

~~~~
var utilityTablePrior = function(){
  var foodValues = [0,1,2,3];
  var timeCostValues = [-0.1, -0.3, -0.6, -1];
  var donut = uniformDraw(foodValues);

  return {donutNorth: donut,
          donutSouth: donut,
          veg: uniformDraw(foodValues),
          noodle: uniformDraw(foodValues),
          timeCost: uniformDraw(timeCostValues)};
};
var alphaPrior = function(){return uniformDraw([.1,1,10,100]);
var world = makeDonutWorld2({big:true, start:[3,1], timeLeft:10});

var posterior = function(observedStateActionSequence){
    return Enumerate( 
  function(){
    var utilityTable = utilityTablePrior();
    var alpha = alphaPrior();
    var agent = makeMDPAgent(utilityTable, alpha, world);
    var act = agent.act;

    // For each observed state-action pair, compute likekihood of action
    map( function(stateAction){
      factor( act(stateAction.state).score( [], stateAction.action) );
    }, observedStateActionSequence )

    return {utilityTable:utilityTable, alpha:alpha};
  });


var observedStateActionSequence = []; // TODO add observed
posterior(observedStateActionSequence.slice(0,1))
posterior(observedStateActionSequence.slice(0,2))
posterior(observedStateActionSequence.slice(0,3))


~~~~

The posterior shows that taking a step towards Donut South can now be explained in terms of a high `timeCost`. If the agent has a low value for $$\alpha$$, then this step to the left is fairly likely even if the agent prefers the Noodle Store or Vegetarian Cafe. So including softmax noise in the inference makes inferences about other parameters closer to the prior. However, once we observe three steps towards Donut South, the inferences about preferences become fairly strong. 

As noted above, it is simple to extend our approach to inference to conditioning on multiple sequences of actions. Consider the two sequences below: [first one goes to DSouth from start. Second one goes to DNorth.]

~~~~

var utilityTablePrior = function(){
  var foodValues = [0,1,2,3];
  var timeCostValues = [-0.1, -0.3, -0.6, -1];
  var donut = uniformDraw(foodValues);

  return {donutNorth: donut,
          donutSouth: donut,
          veg: uniformDraw(foodValues),
          noodle: uniformDraw(foodValues),
          timeCost: uniformDraw(timeCostValues)};
};
var alphaPrior = function(){return uniformDraw([.1,1,10,100]);
var world = makeDonutWorld2({big:true, start:[3,1], timeLeft:10});

var posterior = function(observedStateActionSequence){
  return Enumerate( 
    function(){
      var utilityTable = utilityTablePrior();
      var alpha = alphaPrior();
      var agent = makeMDPAgent(utilityTable, alpha, world);
      var act = agent.act;
      
      map( function(stateAction){
        factor( act(stateAction.state).score( [], stateAction.action) );
      }, observedStateActionSequence )
      
      return {utilityTable:utilityTable, alpha:alpha};
    });


  var observedSequence1 = [];
  var observedSequence2 = [];
  posterior(observedSequence1.concat(observedSequence2))
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
b_i = b_{i-1} | s_i, o_i, a_{i-1}

$$

<!-- b_i(s_i) \propto O(s_i,a_{i-1},o_i)\sum_{s_i \in S}{T(s_{i-1),a_{i-1},s_i)b_{i-1}(s_{i-1})} -->

The posterior can thus be written:

<!--
\verb!(2): ! P(U,\alpha,b_0 | (s,o,a)_{0:n}) \propto P(U, \alpha, b_0) \prod_{i=0}^n P( a_i | s_i, b_i, U, \alpha)
-->







---------------------















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
