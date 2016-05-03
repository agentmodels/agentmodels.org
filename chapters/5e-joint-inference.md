---
layout: chapter
title: Joint inference of biases and preferences II
description: Explaining temptation and pre-commitment using either softmax noise or hyperbolic discounting. 

---

<!-- TODO_daniel: For all posteriors plotted in this chapter, plot the prior on the same axis. For some of the codeboxes the prior is already being plotted on a separate axis. For other codeboxes, the prior is not being plotted and so you need to add a line of code to plot the posterior. -->

## Restaurant Choice: Time-inconsistent vs. optimal MDP agents

Returning to the MDP Restaurant Choice problem, we compare a model that assumes an optimal, non-discounting MDP agent to a model that includes both time-inconsistent and optimal agents. We also consider models that expand the set of preferences the agent can have.

<!-- 2. In the POMDP setting (where the restaurants may be open or closed and the agent can learn this from observation), we do joint inference over preferences, beliefs and discounting behavior. We show that our inference approach can produce multiple explanations for the same behavior and that explanations in terms of beliefs and preferences are more plausible than those involving time-inconsistency. 

As we discussed in Chapter V.1, time-inconsistent agents can produce trajectories on the MDP (full knowledge) version of this scenario that never occur for an optimal agent without noise. 

In our first inference example, we do joint inference over preferences, softmax noise and the discounting behavior of the agent. (We assume for this example that the agent has full knowledge and is not Myopic). We compare the preference inferences [that allow for possibility of time inconsistency] to the earlier inference approach that assumes optimality.
-->
 
### Assume discounting, infer "Naive" or "Sophisticated"
Before making a direct comparison, we demonstrate that we can infer the preferences of time-inconsistent agents from observations of their behavior.

First we condition on the path where the agent moves to Donut North. We call this the Naive path because it is distinctive to the Naive hyperbolic discounter (who is tempted by Donut North on the way to Veg):

~~~~
// draw_naive_path
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['naive'];
print('Observations loaded from library function: \n' 
       + JSON.stringify(observedStateAction) + ' \n');

var path = map(first,observedStateAction);
GridWorld.draw(world, {trajectory:path});
~~~~

For inference, we specialize the approach in the previous <a href="/chapters/5d-joint-inference.html#formalization">chapter</a> for agents in MDPs that are potentially time inconsistent. So we infer $$\nu$$ and $$k$$ (the hyperbolic discounting parameters) but not the initial belief state $$b_0$$. The function `exampleGetPosterior` is a slightly simplified version of the library function we use below.

~~~~
// getPosterior_function

var exampleGetPosterior = function(world, prior, observedStateAction){
  return Enumerate(function () {

    // Sample parameters from prior
    var priorUtility = prior.priorUtility;
    var utilityTable = priorUtility();
    var priorDiscounting = prior.discounting
    var sophisticatedOrNaive = priorDiscounting().sophisticatedOrNaive;

    var priorAlpha = prior.priorAlpha;
    // Create agent with those parameters
    var agent = makeHyperbolicDiscounter(
      { utility   : makeRestaurantUtilityFunction(world, utilityTable),
        alpha     : priorAlpha(), 
        discount  : priorDiscounting().discount,
        sophisticatedOrNaive : sophisticatedOrNaive
      }, world);
    
    var agentAction = agent.act;

    // Condition on observed actions
    map(function (stateAction) {
      var state   = stateAction[0];
      var action  = stateAction[1];
      factor(agentAction(state, 0).score([], action)) ; 
    }, observedStateAction);

    // return parameters and summary statistics
    var vegMinusDonut = sum(utilityTable['Veg']) - sum(utilityTable['Donut N']);

    return {
      utility: utilityTable, 
      sophisticatedOrNaive: discounting.sophisticatedOrNaive,
      discount: discounting.discount, 
      alpha: alpha,
      vegMinusDonut: vegMinusDonut,
    };
  });
};
null;
~~~~

This inference function allows for inference over the softmax parameter ($$\alpha$$ or `alpha`) and the discount constant ($$k$$ or `discount`). For this example, we fix these values so that the agent has low noise ($$\alpha=1000$$) and so $$k=1$$. We also fix the `timeCost` utility to be small and negative and Noodle's utility to be negative. We infer only the agent's utilities and whether they are Naive or Sophisticated.

~~~~
// infer_assume_discounting_naive
// Call to hyperbolic library function and helper display function
///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(priorERP, posteriorERP) {

  var priorUtility = priorERP.MAP().val.utility;
  print('Prior highest-probability utility for Veg: ' + priorUtility['Veg']
	+ '. Donut: ' + priorUtility['Donut N'] + ' \n');

  var posteriorUtility = posteriorERP.MAP().val.utility;
  print('Posterior highest-probability utility for Veg: '
	+ posteriorUtility['Veg'] + '. Donut: ' + posteriorUtility['Donut N']
	+ ' \n');
  
  var getPriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(priorERP, label);
    return Math.exp(erp.score([],x));
  };

  var getPosteriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(posteriorERP, label);
    return Math.exp(erp.score([],x));
  };

  var sophisticationPriorDataTable = map(function(x){
    return {sophisticatedOrNaive: x,
	        probability: getPriorProb({sophisticatedOrNaive: x}),
	        distribution: 'prior'};
  }, ['naive', 'sophisticated']);

  var sophisticationPosteriorDataTable = map(function(x){
    return {sophisticatedOrNaive: x,
	        probability: getPosteriorProb({sophisticatedOrNaive: x}),
	        distribution: 'posterior'};
  }, ['naive', 'sophisticated']);

  var sophisticatedOrNaiveDataTable = append(sophisticationPosteriorDataTable,
                                             sophisticationPriorDataTable);

  viz.bar(sophisticatedOrNaiveDataTable, {groupBy: 'distribution'});

  var vegMinusDonutPriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPriorProb({vegMinusDonut: x}),
	        distribution: 'prior'};
  }, [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]);

  var vegMinusDonutPosteriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPosteriorProb({vegMinusDonut: x}),
	        distribution: 'posterior'};
  }, [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]);

  var vegMinusDonutDataTable = append(vegMinusDonutPriorDataTable,
				                      vegMinusDonutPosteriorDataTable);

  viz.bar(vegMinusDonutDataTable, {groupBy: 'distribution'});
  
  var donutTemptingPriorDataTable = map(function(x){
    return {donutTempting: x,
	        probability: getPriorProb({donutTempting: x}),
	        distribution: 'prior'};
  }, [true, false]);

  var donutTemptingPosteriorDataTable = map(function(x){
    return {donutTempting: x,
	        probability: getPosteriorProb({donutTempting: x}),
	        distribution: 'posterior'};
  }, [true, false]);

  var donutTemptingDataTable = append(donutTemptingPriorDataTable,
				                      donutTemptingPosteriorDataTable);

  viz.bar(donutTemptingDataTable, {groupBy: 'distribution'});
};
  
///

// Prior on agent's utility function: each restaurant has an
// *immediate* utility and a *delayed* utility (which is received after a
// delay of 1).
var priorUtility = function(){
  var utilityValues =  [-10, 0, 10, 20];
  var donut = [uniformDraw(utilityValues), uniformDraw(utilityValues)];
  var veg = [uniformDraw(utilityValues), uniformDraw(utilityValues)];
  return {
    'Donut N' : donut,
    'Donut S' : donut,
    'Veg'     : veg,
    'Noodle'  : [-10, -10],
    'timeCost': -.01
  };
};

var priorDiscounting = function(){ 
  return {
    discount: 1,
    sophisticatedOrNaive: uniformDraw(['naive','sophisticated'])
  };
};
var priorAlpha = function(){return 1000;};
var prior = {
  utility:priorUtility,
  discounting:priorDiscounting,
  alpha:priorAlpha
};

// Get world and observations
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['naive'];
var posterior = getPosterior(world, prior, observedStateAction);

// To get the prior, we condition on the empty list of observations
displayResults(getPosterior(world, prior, []), posterior);
~~~~

We display maximum values and marginal distributions for both the prior and the posterior conditioned on the path shown above. To compute the prior, we simply condition on the empty list of observations.

The first graph shows the distribution over whether the agent is Sophisticated or Naive (labeled `sophisticatedOrNaive`). For the other graphs, we compute summary statistics of the agent's parameters and display the distribution over them. The variable `vegMinusDonut` is the difference in *total* utility between Veg and Donut, ignoring the fact that each restaurant has an *immediate* and *delayed* utility. Inference rules out cases where the total utility is equal (which is most likely in the prior), since the agent would simply go to Donut South in that case. Finally, we introduce a variable `donutTempting`, which is true if the agent prefers Veg to Donut North at the start but reverses this preference when adjacent to Donut North. The prior probability of `donutTempting` is less than $$0.1$$, since it depends on relatively delicate balance of utilities and the discounting behavior. The posterior is closer to $$0.9$$, suggesting (along with the posterior on `sophisticatedOrNaive`) that this is the explanation of the data favored by the model. 

--------

Using the same prior, we condition on the "Sophisticated" path (i.e. the path distinctive to the Sophisticated agent who avoids the temptation of Donut North and takes the long route to Veg):

~~~~
// draw_sophisticated_path
var world = makeRestaurantChoiceMDP();
var path = map(first, restaurantNameToObservationTime11['sophisticated']);
GridWorld.draw(world, {trajectory:path});
~~~~

Here are the results of inference: 

~~~~
// infer_assume_discounting_sophisticated

// Definition of world, prior and inference function is same as above codebox

///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(priorERP, posteriorERP) {

  var priorUtility = priorERP.MAP().val.utility;
  print('Prior highest-probability utility for Veg: ' + priorUtility['Veg']
	+ '. Donut: ' + priorUtility['Donut N'] + ' \n');

  var posteriorUtility = posteriorERP.MAP().val.utility;
  print('Posterior highest-probability utility for Veg: '
	+ posteriorUtility['Veg'] + '. Donut: ' + posteriorUtility['Donut N']
	+ ' \n');
  
  var getPriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(priorERP, label);
    return Math.exp(erp.score([],x));
  };

  var getPosteriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(posteriorERP, label);
    return Math.exp(erp.score([],x));
  };

  var sophisticationPriorDataTable = map(function(x){
    return {sophisticatedOrNaive: x,
	        probability: getPriorProb({sophisticatedOrNaive: x}),
	        distribution: 'prior'};
  }, ['naive', 'sophisticated']);

  var sophisticationPosteriorDataTable = map(function(x){
    return {sophisticatedOrNaive: x,
	        probability: getPosteriorProb({sophisticatedOrNaive: x}),
	        distribution: 'posterior'};
  }, ['naive', 'sophisticated']);

  var sophisticatedOrNaiveDataTable = append(sophisticationPriorDataTable,
					                         sophisticationPosteriorDataTable);

  viz.bar(sophisticatedOrNaiveDataTable, {groupBy: 'distribution'});

  var vegMinusDonutPriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPriorProb({vegMinusDonut: x}),
	        distribution: 'prior'};
  }, [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]);

  var vegMinusDonutPosteriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPosteriorProb({vegMinusDonut: x}),
	        distribution: 'posterior'};
  }, [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]);

  var vegMinusDonutDataTable = append(vegMinusDonutPriorDataTable,
				                      vegMinusDonutPosteriorDataTable);

  viz.bar(vegMinusDonutDataTable, {groupBy: 'distribution'});
  
  var donutTemptingPriorDataTable = map(function(x){
    return {donutTempting: x,
	        probability: getPriorProb({donutTempting: x}),
	        distribution: 'prior'};
  }, [true, false]);

  var donutTemptingPosteriorDataTable = map(function(x){
    return {donutTempting: x,
	        probability: getPosteriorProb({donutTempting: x}),
	        distribution: 'posterior'};
  }, [true, false]);

  var donutTemptingDataTable = append(donutTemptingPriorDataTable,
				                      donutTemptingPosteriorDataTable);

  viz.bar(donutTemptingDataTable, {groupBy: 'distribution'});
};
  
// Prior on agent's utility function
var priorUtility = function(){
  var utilityValues =  [-10, 0, 10, 20];
  var donut = [uniformDraw(utilityValues), uniformDraw(utilityValues)];
  var veg = [uniformDraw(utilityValues), uniformDraw(utilityValues)];
  return {
    'Donut N' : donut,
    'Donut S' : donut,
    'Veg'     : veg,
    'Noodle'  : [-10, -10],
    'timeCost': -.01
  };
};

var priorDiscounting = function(){ 
  return {
    discount: 1,
    sophisticatedOrNaive: uniformDraw(['naive','sophisticated'])
  };
};
var priorAlpha = function(){return 1000;};
var prior = {utility:priorUtility, discounting:priorDiscounting, alpha:priorAlpha};
///

// Get world and observations
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['sophisticated'];
var posterior = getPosterior(world, prior, observedStateAction);
displayResults(getPosterior(world, prior, []), posterior);
~~~~

If the agent goes directly to Veg, then they don't provide information about whether they are Naive or Sophisticated. Using the same prior again, we do inference on this path:

~~~~
// draw_vegDirect_path
var world = makeRestaurantChoiceMDP();
var path = map(first, restaurantNameToObservationTime11['vegDirect']);
GridWorld.draw(world, {trajectory:path});
~~~~

Here are the results of inference: 

~~~~
// infer_assume_discount_vegDirect
// Definition of world, prior and inference function is same as above codebox

///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(priorERP, posteriorERP) {

  var priorUtility = priorERP.MAP().val.utility;
  print('Prior highest-probability utility for Veg: ' + priorUtility['Veg']
	+ '. Donut: ' + priorUtility['Donut N'] + ' \n');

  var posteriorUtility = posteriorERP.MAP().val.utility;
  print('Posterior highest-probability utility for Veg: '
	+ posteriorUtility['Veg'] + '. Donut: ' + posteriorUtility['Donut N']
	+ ' \n');
  
  var getPriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(priorERP, label);
    return Math.exp(erp.score([],x));
  };

  var getPosteriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(posteriorERP, label);
    return Math.exp(erp.score([],x));
  };

  var sophisticationPriorDataTable = map(function(x){
    return {sophisticatedOrNaive: x,
	        probability: getPriorProb({sophisticatedOrNaive: x}),
			distribution: 'prior'};
  }, ['naive', 'sophisticated']);

  var sophisticationPosteriorDataTable = map(function(x){
    return {sophisticatedOrNaive: x,
	        probability: getPosteriorProb({sophisticatedOrNaive: x}),
	        distribution: 'posterior'};
  }, ['naive', 'sophisticated']);

  var sophisticatedOrNaiveDataTable = append(sophisticationPriorDataTable,
					                         sophisticationPosteriorDataTable);

  viz.bar(sophisticatedOrNaiveDataTable, {groupBy: 'distribution'});

  var vegMinusDonutPriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPriorProb({vegMinusDonut: x}),
	        distribution: 'prior'};
  }, [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]);

  var vegMinusDonutPosteriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPosteriorProb({vegMinusDonut: x}),
	        distribution: 'posterior'};
  }, [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]);

  var vegMinusDonutDataTable = append(vegMinusDonutPriorDataTable,
				                      vegMinusDonutPosteriorDataTable);

  viz.bar(vegMinusDonutDataTable, {groupBy: 'distribution'});

  
  var donutTemptingPriorDataTable = map(function(x){
    return {donutTempting: x,
	        probability: getPriorProb({donutTempting: x}),
			distribution: 'prior'};
  }, [true, false]);

  var donutTemptingPosteriorDataTable = map(function(x){
    return {donutTempting: x,
	        probability: getPosteriorProb({donutTempting: x}),
			distribution: 'posterior'};
  }, [true, false]);

  var donutTemptingDataTable = append(donutTemptingPriorDataTable,
				                      donutTemptingPosteriorDataTable);

  viz.bar(donutTemptingDataTable, {groupBy: 'distribution'});
};

// Prior on agent's utility function
var priorUtility = function(){
  var utilityValues =  [-10, 0, 10, 20];
  var donut = [uniformDraw(utilityValues), uniformDraw(utilityValues)];
  var veg = [uniformDraw(utilityValues), uniformDraw(utilityValues)];
  return {
    'Donut N' : donut,
    'Donut S' : donut,
    'Veg'     : veg,
    'Noodle'  : [-10, -10],
    'timeCost': -.01
  };
};

var priorDiscounting = function(){ 
  return {
    discount: 1,
    sophisticatedOrNaive: uniformDraw(['naive','sophisticated'])
  };
};
var priorAlpha = function(){return 1000;};
var prior = {utility:priorUtility, discounting:priorDiscounting, alpha:priorAlpha};
///

// Get world and observations
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['vegDirect'];
var posterior = getPosterior(world, prior, observedStateAction);
displayResults(getPosterior(world, prior, []), posterior);
~~~~

<br>

---------

### Assume non-discounting, infer preferences and softmax
We want to compare a model that assumes an optimal MDP agent with one that allows for time-inconsistency. We first show the inferences by the model that assumes optimality. This model can only explain the anomalous Naive and Sophisticated paths in terms of softmax noise (lower values for $$\alpha$$). We display the prior and posteriors for both the Naive and Sophisticated paths. 

~~~~
// infer_assume_optimal_naive_sophisticated
///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(priorERP, posteriorERP) {

  var priorUtility = priorERP.MAP().val.utility;
  print('Prior highest-probability utility for Veg: ' + priorUtility['Veg']
	+ '. Donut: ' + priorUtility['Donut N'] + ' \n');

  var posteriorUtility = posteriorERP.MAP().val.utility;
  print('Posterior highest-probability utility for Veg: '
	+ posteriorUtility['Veg'] + '. Donut: ' + posteriorUtility['Donut N']
	+ ' \n');
  
  var getPriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(priorERP, label);
    return Math.exp(erp.score([],x));
  };

  var getPosteriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(posteriorERP, label);
    return Math.exp(erp.score([],x));
  };

  var vegMinusDonutPriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPriorProb({vegMinusDonut: x}),
			distribution: 'prior'};
  }, [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]);

  var vegMinusDonutPosteriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPosteriorProb({vegMinusDonut: x}),
			distribution: 'posterior'};
  }, [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]);

  var vegMinusDonutDataTable = append(vegMinusDonutPriorDataTable,
				                      vegMinusDonutPosteriorDataTable);

  viz.bar(vegMinusDonutDataTable, {groupBy: 'distribution'});
  
  var alphaPriorDataTable = map(function(x){
    return {alpha: x,
	        probability: getPriorProb({alpha: x}),
	        distribution: 'prior'};
  }, [0.1, 10, 100, 1000]);

  var alphaPosteriorDataTable = map(function(x){
    return {alpha: x,
	        probability: getPosteriorProb({alpha: x}),
	        distribution: 'posterior'};
  }, [0.1, 10, 100, 1000]);

  var alphaDataTable = append(alphaPriorDataTable,
			                  alphaPosteriorDataTable);

  viz.bar(alphaDataTable, {groupBy: 'distribution'});
};
///

// Prior on agent's utility function
var priorUtility = function(){
  var utilityValues = [-10,0,10,20,30,40];
  // with no discounting, delayed utilities are ommitted
  var donut = [uniformDraw(utilityValues), 0];
  var veg = [uniformDraw(utilityValues), 0];
  return {
    'Donut N' : donut,
    'Donut S' : donut,
    'Veg'     : veg,
    'Noodle'  : [-10, -10],
    'timeCost': -.01
  };
};

// We assume no discounting (so *sophisticated* has no effect here)
var priorDiscounting = function(){
  return {
    discount: 0,
    sophisticatedOrNaive: 'sophisticated'
  };
};

var priorAlpha = function(){return uniformDraw([0.1, 10, 100, 1000]);};
var prior = {utility:priorUtility, discounting:priorDiscounting, alpha:priorAlpha};

// Get world and observations
var world = makeRestaurantChoiceMDP();

print('Prior and posterior after observing Naive path');

var observedStateActionNaive = restaurantNameToObservationTime11['naive'];
var posteriorNaive = getPosterior(world, prior, observedStateActionNaive);
displayResults(getPosterior(world, prior, []), posteriorNaive);

print('Prior and posterior after observing Sophisticated path');

var observedStateActionSoph = restaurantNameToObservationTime11['sophisticated'];
var posteriorSophisticated = getPosterior(world, prior, observedStateActionSoph);
displayResults(getPosterior(world, prior, []), posteriorSophisticated);
~~~~

The graphs show two important results:

1. For the Naive path, the agent is inferred to prefer Donut, while for the Sophisticated path, Veg is inferred. In both cases, the inference fits with where the agent ends up. 

2. High values for $$\alpha$$ are ruled out in each case, showing that the model explains the behavior in terms of noise. 

What happens if we observe the agent taking the Naive path *repeatedly*? While noise is needed to explain the agent's path, too much noise is inconsistent with taking an identical path repeatedly. This is confirmed in the results below:

~~~~
// infer_assume_optimal_naive_three_times

// Prior is same as above:
///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(priorERP, posteriorERP) {

  var priorUtility = priorERP.MAP().val.utility;
  print('Prior highest-probability utility for Veg: ' + priorUtility['Veg']
	+ '. Donut: ' + priorUtility['Donut N'] + ' \n');

  var posteriorUtility = posteriorERP.MAP().val.utility;
  print('Posterior highest-probability utility for Veg: '
	+ posteriorUtility['Veg'] + '. Donut: ' + posteriorUtility['Donut N']
	+ ' \n');
  
  var getPriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(priorERP, label);
    return Math.exp(erp.score([],x));
  };

  var getPosteriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(posteriorERP, label);
    return Math.exp(erp.score([],x));
  };

  var vegMinusDonutPriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPriorProb({vegMinusDonut: x}),
			distribution: 'prior'};
  }, [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]);

  var vegMinusDonutPosteriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPosteriorProb({vegMinusDonut: x}),
			distribution: 'posterior'};
  }, [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]);

  var vegMinusDonutDataTable = append(vegMinusDonutPriorDataTable,
				                      vegMinusDonutPosteriorDataTable);

  viz.bar(vegMinusDonutDataTable, {groupBy: 'distribution'});
  
  var alphaPriorDataTable = map(function(x){
    return {alpha: x,
	        probability: getPriorProb({alpha: x}),
	        distribution: 'prior'};
  }, [0.1, 10, 100, 1000]);

  var alphaPosteriorDataTable = map(function(x){
    return {alpha: x,
	        probability: getPosteriorProb({alpha: x}),
	        distribution: 'posterior'};
  }, [0.1, 10, 100, 1000]);

  var alphaDataTable = append(alphaPriorDataTable,
			                  alphaPosteriorDataTable);

  viz.bar(alphaDataTable, {groupBy: 'distribution'});
};
  
// Prior on agent's utility function
var priorUtility = function(){
  var utilityValues = [-10,0,10,20,30,40];
  // with no discounting, delayed utilities are ommitted
  var donut = [uniformDraw(utilityValues), 0];
  var veg = [uniformDraw(utilityValues), 0];
  return {
    'Donut N' : donut,
    'Donut S' : donut,
    'Veg'     : veg,
    'Noodle'  : [-10, -10],
    'timeCost': -.01
  };
};

// We assume no discounting (so *sophisticated* has no effect here)
var priorDiscounting = function(){
  return {
    discount: 0,
    sophisticatedOrNaive: 'sophisticated'
  };
};

var priorAlpha = function(){return uniformDraw([0.1, 10, 100, 1000]);};
var prior = {utility:priorUtility, discounting:priorDiscounting, alpha:priorAlpha};
///

// Get world and observations
var world = makeRestaurantChoiceMDP();
var observedStateActionNaive = restaurantNameToObservationTime11['naive'];
var numberRepeats = 2; // with 2 repeats, we condition a total of 3 times
var posteriorNaive = getPosterior(world, prior, observedStateActionNaive,
                                  numberRepeats);
print('Prior and posterior after conditioning 3 times on Naive path');
displayResults(getPosterior(world, prior, []), posteriorNaive);
~~~~

<br>

--------

### Model that includes discounting: jointly infer discounting, preferences, softmax noise

Our inference model now has the optimal agent as a special case but also includes time-inconsistent agents. This model jointly infers the discounting behavior, the agent's utilities and the softmax noise. 

We show two different posteriors. The first is after conditioning on the Naive path (as above). In the second, we imagine that we have observed the agent taking the same path on multiple occasions (three times) and we condition on this. 

~~~~
// infer_joint_model_naive

///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(priorERP, posteriorERP) {

  var priorUtility = priorERP.MAP().val.utility;
  print('Prior highest-probability utility for Veg: ' + priorUtility['Veg']
	+ '. Donut: ' + priorUtility['Donut N'] + ' \n');

  var posteriorUtility = posteriorERP.MAP().val.utility;
  print('Posterior highest-probability utility for Veg: '
	+ posteriorUtility['Veg'] + '. Donut: ' + posteriorUtility['Donut N']
	+ ' \n');
  
  var getPriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(priorERP, label);
    return Math.exp(erp.score([],x));
  };

  var getPosteriorProb = function(x){
    var label = _.keys(x)[0];
    var erp = getMarginalObject(posteriorERP, label);
    return Math.exp(erp.score([],x));
  };

  var sophisticationPriorDataTable = map(function(x){
    return {sophisticatedOrNaive: x,
	        probability: getPriorProb({sophisticatedOrNaive: x}),
	        distribution: 'prior'};
  }, ['naive', 'sophisticated']);

  var sophisticationPosteriorDataTable = map(function(x){
    return {sophisticatedOrNaive: x,
	        probability: getPosteriorProb({sophisticatedOrNaive: x}),
	        distribution: 'posterior'};
  }, ['naive', 'sophisticated']);

  var sophisticatedOrNaiveDataTable = append(sophisticationPosteriorDataTable,
                                             sophisticationPriorDataTable);

  viz.bar(sophisticatedOrNaiveDataTable, {groupBy: 'distribution'});

  var vegMinusDonutPriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPriorProb({vegMinusDonut: x}),
	        distribution: 'prior'};
  }, [-10, 0, 10, 20, 30, 40, 50, 60, 70]);

  var vegMinusDonutPosteriorDataTable = map(function(x){
    return {vegMinusDonut: x,
	        probability: getPosteriorProb({vegMinusDonut: x}),
	        distribution: 'posterior'};
  }, [-10, 0, 10, 20, 30, 40, 50, 60, 70]);

  var vegMinusDonutDataTable = append(vegMinusDonutPriorDataTable,
				                      vegMinusDonutPosteriorDataTable);

  viz.bar(vegMinusDonutDataTable, {groupBy: 'distribution'});
  
  var donutTemptingPriorDataTable = map(function(x){
    return {donutTempting: x,
	        probability: getPriorProb({donutTempting: x}),
	        distribution: 'prior'};
  }, [true, false]);

  var donutTemptingPosteriorDataTable = map(function(x){
    return {donutTempting: x,
	        probability: getPosteriorProb({donutTempting: x}),
	        distribution: 'posterior'};
  }, [true, false]);

  var donutTemptingDataTable = append(donutTemptingPriorDataTable,
				                      donutTemptingPosteriorDataTable);

  viz.bar(donutTemptingDataTable, {groupBy: 'distribution'});

  var alphaPriorDataTable = map(function(x){
    return {alpha: x,
	        probability: getPriorProb({alpha: x}),
	        distribution: 'prior'};
  }, [0.1, 10, 1000]);

  var alphaPosteriorDataTable = map(function(x){
    return {alpha: x,
	        probability: getPosteriorProb({alpha: x}),
	        distribution: 'posterior'};
  }, [0.1, 10, 1000]);

  var alphaDataTable = append(alphaPriorDataTable,
			                  alphaPosteriorDataTable);

  viz.bar(alphaDataTable, {groupBy: 'distribution'});
};

///

// Prior on agent's utility function. We fix the delayed utilities
// to make inference faster
  var priorUtility = function(){
    var utilityValues =  [-10, 0, 10, 20, 30];
    var donut = [uniformDraw(utilityValues), -10];
    var veg = [uniformDraw(utilityValues), 20];
    return {
      'Donut N' : donut,
      'Donut S' : donut,
      'Veg'     : veg,
      'Noodle'  : [-10, -10],
      'timeCost': -.01
    };
  };

  var priorDiscounting = function(){ 
    return {
      discount: uniformDraw([0,1]),
      sophisticatedOrNaive: uniformDraw(['naive','sophisticated'])
    };
  };
  var priorAlpha = function(){return uniformDraw([.1, 10, 1000]);};
var prior = {utility:priorUtility, 
             discounting:priorDiscounting, 
             alpha:priorAlpha};

// Get world and observations
var world = makeRestaurantChoiceMDP(); //  makeRestaurantChoiceMDP({noReverse:false});

var observedStateAction = restaurantNameToObservationTime11['naive'];
var posterior = getPosterior(world, prior, observedStateAction);
print('Prior and posterior after observing Naive path');
displayResults(getPosterior(world, prior, []), posterior);

print('Prior and posterior after observing Naive path three times');
var numberRepeats = 2;
displayResults(getPosterior(world, prior, []),
               getPosterior(world, prior, observedStateAction, numberRepeats));
~~~~

Conditioning on the Naive path once, the probabilities of the agent being Naive and of `donutTempting` both go up. However, the probability of high softmax noise also goes up. In terms of preferences, we rule out a strong preference for Veg and slightly reduce a preference for Donut. So if the agent were Naive, tempted by Donut and with very low noise, our inference would not place most of the posterior on this explanation. There are two reasons for this. First, this agent is unlikely in the prior. Second, the explanation of the behavior in terms of noise is plausible. (In our Gridworld setup, we don't allow the agent to backtrack to the previous state. This means there are few cases where a softmax noisy agent would behavior differently than a low noise one.). Conditioning on the same Naive path three times makes the explanation in terms of noise much less plausible: the agent would makes the same "mistake" three times and makes no other mistakes. (The results for the Sophisticated path are similar.)

In summary, if we observe the agent repeatedly take the Naive path, the "Optimal Model" explains this in terms of a preference for Donut and significant softmax noise (explaining why the agent takes Donut North over Donut South). The "Discounting Model" is similar to the Optimal Model when it observes the Naive path *once*. However, observing it multiple times, it infers that the agent has low noise and an overall preference for Veg. 

<br>

------

### Preferences for the two Donut Store branches can vary
Another explanation of the Naive path is that the agent has a preference for the "Donut N" branch of the Donut Store over the "Donut S" branch. Maybe this branch is better run or has more space. If we add this to our set of possible preferences, inference changes significantly.

To speed up inference, we use a fixed assumption that the agent is Naive. There are three explanations of the agent's path:

(1). Softmax noise: measured by $$\alpha$$
(2). The agent is Naive and tempted by Donut: measured by `discount` and `donutTempting`
(3). The agent prefers Donut N to Donut S: measured by `donutNGreaterDonutS` (i.e. Donut N's utility is greater than Donut S's).

These three can also be combined to explain the behavior. 

~~~~
// infer_joint_two_donut_naive
///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(erp, label){
  if (label){print('Display: ' + label)}
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg'] 
        +'. Donut N: ' + utility['Donut N'] +
        +'. Donut S: ' + utility['Donut S']);
  var alphaPrint = Enumerate(function(){
    return {alpha: JSON.stringify(sample(erp).alpha) };
  });                          
  viz.auto(alphaPrint);
  viz.auto(getMarginalObject(erp, 'donutTempting'));
  viz.auto(getMarginalObject(erp, 'discount'));
  viz.auto(getMarginalObject(erp, 'donutNGreaterDonutS'));
};
///

// Prior on agent's utility function
var priorUtility = function(){
  var utilityValues =  [-10, 0, 10, 20];
  
  return {
    'Donut N' : [uniformDraw(utilityValues), -10],
    'Donut S' : [uniformDraw(utilityValues), -10],
    'Veg'     : [20, uniformDraw(utilityValues)],
    'Noodle'  : [-10, -10],
    'timeCost': -.01
  };
};

var priorDiscounting = function(){ 
  return {
    discount: uniformDraw([0,1]),
    sophisticatedOrNaive: 'naive'
  };
};
var priorAlpha = function(){return uniformDraw([.1, 100, 1000]);};
var prior = {utility:priorUtility, discounting:priorDiscounting, alpha:priorAlpha};

// Get world and observations
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['naive'];
var posterior = getPosterior(world, prior, observedStateAction);
displayResults(getPosterior(world, prior, []), 'Prior distribution')
displayResults(posterior, 'Posterior on Naive path');
~~~~

The explanation in terms of Donut North being preferred does well in the posterior. This is because the discounting explanation (even assuming the agent is Naive) is unlikely a priori (due to our simple uniform priors on utilities and discounting). While high noise is more plausible a priori, the noise explanation still needs to posit a low probability series of events. 

We see a similar result if we enrich the set of possible utilities for the Sophisticated path. This time, we allow the `timeCost`, i.e. the cost for taking a single timestep, to be positive. This means the agent prefers to spend as much time as possible moving around before reaching a restaurant. Here are the results:

Observe the sophisticated path with possibly positive timeCost:

~~~~
// infer_joint_timecost_sophisticated
///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(erp, label){
  if (label){print('Display: ' + label)}
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg'] 
        +'. Donut: ' + utility['Donut N'] +
        +'. TimeCost: ' + utility['timeCost']);
  var alphaPrint = Enumerate(function(){
    return {alpha: JSON.stringify(sample(erp).alpha) };
  });                          
  viz.auto(alphaPrint);
  viz.auto(getMarginalObject(erp, 'donutTempting'));
  viz.auto(getMarginalObject(erp, 'discount'));
   var timePrint = Enumerate(function(){
    return {timeCost: JSON.stringify(sample(erp).utility.timeCost) };
  });   
  viz.auto(timePrint);
};
///


// Prior on agent's utility function
var priorUtility = function(){
  var utilityValues =  [-10, 0, 10, 20, 30];
  var donut = [uniformDraw(utilityValues), -10]
  var veg = [uniformDraw(utilityValues), 20];
  return {
    'Donut N' : donut,
    'Donut S' : donut,
    'Veg'     : veg,
    'Noodle'  : [-10, -10],
    'timeCost': uniformDraw([-0.01, 0.1, 1])
  };
};

var priorDiscounting = function(){ 
  return {
    discount: uniformDraw([0,1]),
    sophisticatedOrNaive: 'sophisticated'
  };
};
var priorAlpha = function(){return uniformDraw([0.1, 100, 1000]);};
var prior = {utility:priorUtility, discounting:priorDiscounting, alpha:priorAlpha};

// Get world and observations
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['sophisticated'];
var posterior = getPosterior(world, prior, observedStateAction);
displayResults(getPosterior(world, prior, []), 'Prior distribution')
displayResults(posterior, 'Posterior on Sophisticated Path');
~~~~                             


