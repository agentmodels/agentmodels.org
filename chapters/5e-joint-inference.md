---
layout: chapter
title: Joint inference of biases and preferences II
description: Explaining temptation and pre-commitment using either softmax noise or hyperbolic discounting. 

---


## Restaurant Choice Inference

Returning to the MDP Restaurant Choice problem, we compare a model that assumes an optimal, non-discounting MDP agent to a model that includes both time-inconsistent and optimal agents. We also consider models that allow a richer set of preferences. 

<!-- 2. In the POMDP setting (where the restaurants may be open or closed and the agent can learn this from observation), we do joint inference over preferences, beliefs and discounting behavior. We show that our inference approach can produce multiple explanations for the same behavior and that explanations in terms of beliefs and preferences are more plausible than those involving time-inconsistency. 

As we discussed in Chapter V.1, time-inconsistent agents can produce trajectories on the MDP (full knowledge) version of this scenario that never occur for an optimal agent without noise. 

In our first inference example, we do joint inference over preferences, softmax noise and the discounting behavior of the agent. (We assume for this example that the agent has full knowledge and is not Myopic). We compare the preference inferences [that allow for possibility of time inconsistency] to the earlier inference approach that assumes optimality.
-->

### Example 1: Time-inconsistent vs. optimal MDP agents
This example compares a model that assumes an optimal agent (and just infers their preferences and softmax noise) to a model that also allows for sub-optimal, time-inconsistent agents. Before making a direct comparison, we demonstrate that we can infer the preferences of time-inconsistent agents from observations of their behavior.

#### Assume discounting, infer "Naive" or "Sophisticated"
First we condition on the path where the agent moves to Donut North. We call this the Naive path because it is distinctive to the Naive hyperbolic discounter (tempted by Donut North on the way to Veg):

~~~~
// draw_naive_path
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['naive'];
print('Observations loaded from library function: \n' 
       + JSON.stringify(observedStateAction) + ' \n');

var path = map(first,observedStateAction);
GridWorld.draw(world, {trajectory:path});
~~~~

For inference, we specialize the approach above for agents in MDPs that are potentially time inconsistent. So we infer $$\nu$$ and $$k$$ (the hyperbolic discounting parameters) but not the initial belief state $$b_0$$. The function `exampleGetPosterior` is a slightly simplified version of the library function we use below.

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

var displayResults = function(erp, label){
  if (label){print('Display: ' + label)}
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg'] + 
  '. Donut: ' + utility['Donut N'] + ' \n')
  viz.auto(getMarginalObject(erp,'sophisticatedOrNaive'));
  viz.auto(getMarginalObject(erp,'vegMinusDonut'));
  viz.auto(getMarginalObject(erp,'donutTempting'));
};
///

// Prior on agent's utility function: each restaurant has an
// *immediate* utility and a *delayed* utility (which is received after a delay of 1). 
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

// Get world and observations
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['naive'];
var posterior = getPosterior(world, prior, observedStateAction);

// To get the prior, we condition on the empty list of obserations
displayResults(getPosterior(world, prior, []), 'Prior distribution');
displayResults(posterior, 'Posterior distribution');
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

var displayResults = function(erp, label){
  if (label){print('Display: ' + label)}
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg'] + 
  '. Donut: ' + utility['Donut N'] + ' \n')
  viz.auto(getMarginalObject(erp,'sophisticatedOrNaive'));
  viz.auto(getMarginalObject(erp,'vegMinusDonut'));
  viz.auto(getMarginalObject(erp,'donutTempting'));
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
displayResults(posterior, 'Posterior distribution conditioning on Sophisticated path: ');
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

var displayResults = function(erp, label){
  if (label){print('Display: ' + label)}
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg'] + 
  '. Donut: ' + utility['Donut N'] + ' \n')
  viz.auto(getMarginalObject(erp,'sophisticatedOrNaive'));
  viz.auto(getMarginalObject(erp,'vegMinusDonut'));
  viz.auto(getMarginalObject(erp,'donutTempting'));
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
displayResults(posterior, 'Posterior distribution conditioning on VegDirect path');
~~~~

#### Assume non-discounting, infer preferences and softmax
We want to compare a model that assumes an optimal MDP agent with one that allows for time-inconsistency. We first show the inferences by the model that assumes optimality. This model can only explain the anomalous Naive and Sophisticated paths in terms of softmax noise (lower values for $$\alpha$$). We display the prior and posteriors for both the Naive and Sophisticated paths. 

~~~~
// infer_assume_optimal_naive_sophisticated
///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(erp, label){
  if (label){print('Display: ' + label)}
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg'] + 
  '. Donut: ' + utility['Donut N'] + ' \n')
  viz.auto(getMarginalObject(erp,'vegMinusDonut'));
  var alphaPrint = Enumerate(function(){
    return {alpha: JSON.stringify(sample(erp).alpha) };
  });                          
  viz.auto(alphaPrint);
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
var observedStateActionNaive = restaurantNameToObservationTime11['naive'];
var posteriorNaive = getPosterior(world, prior, observedStateActionNaive);
displayResults(getPosterior(world, prior, []), 'Prior distribution')
displayResults( posteriorNaive, 'Posterior distribution on Naive path')

var observedStateActionSoph = restaurantNameToObservationTime11['sophisticated'];
var posteriorSophisticated = getPosterior(world, prior, observedStateActionSoph);
displayResults( posteriorSophisticated,
'Posterior distribution on Sophisticated path')
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

var displayResults = function(erp, label){
  if (label){print('Display: ' + label)}
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg'] + 
  '. Donut: ' + utility['Donut N'] + ' \n')
  viz.auto(getMarginalObject(erp,'vegMinusDonut'));
  var alphaPrint = Enumerate(function(){
    return {alpha: JSON.stringify(sample(erp).alpha) };
  });                          
  viz.auto(alphaPrint);
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
var posteriorNaive = getPosterior(world, prior, observedStateActionNaive, numberRepeats);
displayResults( posteriorNaive, 'Posterior on conditioning 3 times on Naive path')
~~~~



#### Model that includes discounting: jointly infer discounting, preferences, softmax noise

Our inference model now has the optimal agent as a special case but also includes time-inconsistent agents. This model jointly infers the discounting behavior, the agent's utilities and the softmax noise. 

We show two different posteriors. The first is after conditioning on the Naive path (as above). In the second, we imagine that we have observed the agent taking the same path on multiple occasions (three times) and we condition on this. 

~~~~
// infer_joint_model_naive

///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(erp, label){
  if (label){print('Display: ' + label)}
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg'] + 
  '. Donut: ' + utility['Donut N'] + ' \n')
  viz.auto(getMarginalObject(erp,'sophisticatedOrNaive'));
  viz.auto(getMarginalObject(erp,'vegMinusDonut'));
  viz.auto(getMarginalObject(erp,'donutTempting'));
  var alphaPrint = Enumerate(function(){
    return {alpha: JSON.stringify(sample(erp).alpha) };
  });                          
  viz.auto(alphaPrint);
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
displayResults(getPosterior(world, prior, []), 'Prior distribution');
displayResults(posterior, 'Posterior on Naive path');
var numberRepeats = 2;
displayResults(getPosterior(world, prior, observedStateAction, numberRepeats), 'Posterior on three repeats of Naive path')

~~~~

Conditioning on the Naive path once, the probabilities of the agent being Naive and of `donutTempting` both go up. However, the probability of high softmax noise also goes up. In terms of preferences, we rule out a strong preference for Veg and slightly reduce a preference for Donut. So if the agent were Naive, tempted by Donut and with very low noise, our inference would not place most of the posterior on this explanation. There are two reasons for this. First, this agent is unlikely in the prior. Second, the explanation of the behavior in terms of noise is plausible. (In our Gridworld setup, we don't allow the agent to backtrack to the previous state. This means there are few cases where a softmax noisy agent would behavior differently than a low noise one.). Conditioning on the same Naive path three times makes the explanation in terms of noise much less plausible: the agent would makes the same "mistake" three times and makes no other mistakes. (The results for the Sophisticated path are similar.)

In summary, if we observe the agent repeatedly take the Naive path, the "Optimal Model" explains this in terms of a preference for Donut and significant softmax noise (explaining why the agent takes Donut North over Donut South). The "Discounting Model" is similar to the Optimal Model when it observes the Naive path *once*. However, observing it multiple times, it infers that the agent has low noise and an overall preference for Veg. 


#### Preferences for the two Donut Store branches can vary
Another explanation of the Naive path is that the agent has a preference for the "Donut N" branch of the Donut Store over the "Donut S" branch. Maybe this branch is better run or has more space. If we add this to our set of possible preferences, inference changes significantly.

To speed up inference, we use a fixed assumption that the agent is Naive. There are three explanations of the agent's path:

1. Softmax noise: measured by $$\alpha$$
2. The agent is Naive and tempted by Donut: measured by `discount` and `donutTempting`
3. The agent prefers Donut N to Donut S: measured by `donutNGreaterDonutS` (i.e. Donut N's utility is greater than Donut S's).

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


