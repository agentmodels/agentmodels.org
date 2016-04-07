---
layout: chapter
title: Joint inference of biases, beliefs, and preferences
description: Assuming the agent performs optimally can lead to mistakes in inference. Show that we can do joint inference over large space of agents. 

---

### Introduction
In the opening [chapter](/chapters/5-biases-intro) of this section, we argued that human behavior in sequential decision problems won't always conform to optimal solving of (PO)MDPs. So if our goal is learning about human beliefs and preferences from their actions (i.e. Inverse Reinforcement Learning), then we might do better with more realistic generative models for human behavior. This chapter explores how adding time inconsistency and myopic planning to agent models affects inference of preferences.

If human behavior in some decision problem always conforms exactly to a particular sub-optimal planning model, then it would be surprising if using the true generative model for inference did not help with accurate recovery of preferences. Biases will only affect some of the humans some of the time. In a narrow domain, experts can learn to avoid biases and they can use specialized approximation algorithms that achieve near-optimal performance in the domain. So our approach is to do *joint inference* over preferences, beliefs and biases and cognitive bounds. If the agent's behavior is consistent with optimal (PO)MDP solving, we will infer this fact and infer preferences accordingly. On the other hand, if there's evidence of biases, this will alter inferences about preferences. We test our approach by comparing to a model that has a fixed assumption of optimality. We show that in simple, intuitive decision problems, assuming optimality leads to mistaken inferences about preferences.

As we discussed in Chapter IV, the identifiability of preferences is a ubiquitous issue in IRL. Our approach, which does inference over a broader space of agents (with different combinations of biases), makes identification from a particular decision problem less likely in general. Yet the lack of identifiability of preferences is not something that undermines our approach. For some decision problems, the best an inference system can do is rule out preferences that are inconsistent with the behavior and accurately maintain posterior uncertainty over those that are consistent. Some of the examples below provide behavior that is ambiguous about preferences in this way. Yet we also show simple examples in which biases and bounds *can* be identified. 


### Formalization of Joint Inference
We formalize joint inference over beliefs, preferences and biases by extending the approach developing in Chapter IV. In Equation (2) of that chapter, an agent was characterized by parameters $$  \left\langle U, \alpha, b_0 \right\rangle$$. To include the possibility of time-inconsistent and Myopic agents, an agent $$\theta$$ is now characterized by a tuple of parameters as follows:

$$
\theta = \left\langle U, \alpha, b_0, k, \nu, C \right\rangle
$$

where:

- $$U$$ is the utilty function

- $$\alpha$$ is the softmax noise parameter

- $$b_0$$ is the agent's belief (or prior) over the initial state


- $$k \geq 0$$ is the constant for hyperbolic discounting function $$1/(1+kd)$$

- $$\nu$$ is an indicator for Naive or Sophisticated hyperbolic discounting

- $$C \in [1,\infty]$$ is the integer cutoff point for Myopic Exploration. 

As in Equation (2), we condition on state-action-observation triples:

$$
P(\theta \vert (s,o,a)_{0:n}) \propto P( (s,o,a)_{0:n} \vert \theta)P(\theta)
$$

We obtain a factorized form in exactly the same way as in Equation (2), i.e. we generate the sequence $$b_i$$ from $$i=0$$ to $$i=n$$ of agent beliefs:

$$
P(\theta \vert (s,o,a)_{0:n}) \propto 
P(\theta) \prod_{i=0}^n P( a_i \vert s_i, b_i, U, \alpha, k, \nu, C )
$$

The likelihood term on the RHS of this equation is simply the softmax probability that the agent with given parameters chooses $$a_i$$ in state $$s_i$$. This equation for inference does not make use of the *delay* indices used by time-inconsistent and Myopic agents. This is because the delays figure only in their internal simulations. In order to compute the likelihood the agent takes an action, we don't need to keep track of delay values. 


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
      { utility   : makeRestaurantUtilityMDP(world, utilityTable),
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
  viz.vegaPrint(getMarginalObject(erp,'sophisticatedOrNaive'));
  viz.vegaPrint(getMarginalObject(erp,'vegMinusDonut'));
  viz.vegaPrint(getMarginalObject(erp,'donutTempting'));
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
  viz.vegaPrint(getMarginalObject(erp,'sophisticatedOrNaive'));
  viz.vegaPrint(getMarginalObject(erp,'vegMinusDonut'));
  viz.vegaPrint(getMarginalObject(erp,'donutTempting'));
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
  viz.vegaPrint(getMarginalObject(erp,'sophisticatedOrNaive'));
  viz.vegaPrint(getMarginalObject(erp,'vegMinusDonut'));
  viz.vegaPrint(getMarginalObject(erp,'donutTempting'));
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
  viz.vegaPrint(getMarginalObject(erp,'vegMinusDonut'));
  var alphaPrint = Enumerate(function(){
    return {alpha: JSON.stringify(sample(erp).alpha) };
  });                          
  viz.vegaPrint(alphaPrint);
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
  viz.vegaPrint(getMarginalObject(erp,'vegMinusDonut'));
  var alphaPrint = Enumerate(function(){
    return {alpha: JSON.stringify(sample(erp).alpha) };
  });                          
  viz.vegaPrint(alphaPrint);
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
  viz.vegaPrint(getMarginalObject(erp,'sophisticatedOrNaive'));
  viz.vegaPrint(getMarginalObject(erp,'vegMinusDonut'));
  viz.vegaPrint(getMarginalObject(erp,'donutTempting'));
  var alphaPrint = Enumerate(function(){
    return {alpha: JSON.stringify(sample(erp).alpha) };
  });                          
  viz.vegaPrint(alphaPrint);
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
  viz.vegaPrint(alphaPrint);
  viz.vegaPrint(getMarginalObject(erp, 'donutTempting'));
  viz.vegaPrint(getMarginalObject(erp, 'discount'));
  viz.vegaPrint(getMarginalObject(erp, 'donutNGreaterDonutS'));
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
  viz.vegaPrint(alphaPrint);
  viz.vegaPrint(getMarginalObject(erp, 'donutTempting'));
  viz.vegaPrint(getMarginalObject(erp, 'discount'));
   var timePrint = Enumerate(function(){
    return {timeCost: JSON.stringify(sample(erp).utility.timeCost) };
  });   
  viz.vegaPrint(timePrint);
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

-----------

------------


## Naive/Soph/Neutral examples for Restaurant Choice Gridworld

- Joint inference for Naive and Soph examples:

In Naive, we consider false belief about Donut South being closed, preference for Donut North over south (unlikely in the prior) and the discounting explanation. Show multimodal inference. Discuss identification issues and Bayes Occam.

For Soph, we compare false belief the Noodle is open, a positive timeCost (which is has a low prior probability), and the Soph explanation. We could mention the experiment showing that when belief / preference explanations were possible, people tended to prefer them over HD explanations (though they did generate HD explanations spontaneously).

- Big inference example (maybe in later chapter): use HMC and do inference oiver cts params.

## Learning preferences of procrastinators

The Procrastination Problem from Chapter V.1. illustrates how agents with identical preferences can deviate *systematically* in their behavior due to time inconsistency. Suppose two agents care equally about finishing the task and assign the same cost to doing the hard work. The optimal agent will complete the task immediately. The Naive hyperbolic discounter will delay every day until the deadline, which could be (say) thirty days away!

This kind of systematic deviation between agents is also significant for inferring preferences. We consider the problem of *online* inference, where we observe the agent's behavior each day and produce an estimate of their preferences. Suppose the agent has a deadline $$T$$ days into the future and leaves the work till the last day. As we discussed earlier, this is just the kind of behavior we see in people every day -- and so is a good test for a model of inference. We compare the online inferences of two models. The *Optimal Model* assumes the agent is time-consistent with softmax parameter $$\alpha$$. The *Possibly Discounting* model includes both optimal and Naive hyperbolic discounting agents in the prior.

For each model, we compute posteriors for the agent's parameters after observing the agent's choice at each timestep. We set $$T=10$$. So the observed actions are `["wait", "wait", "wait", ... , "work"]`, where `"work"` is the final action. We fix the utilities for doing the work (the `workCost` or $$-w$$) and for delaying the work (the `waitCost` or $$-\epsilon$$). We learn the following parameters:

- The reward for doing the task: $$R$$ or `reward`
- The agent's softmax parameter $$\alpha$$
- The agent's discount rate (for the Possibly Discounting model): $$k$$ or `discount`

For each parameter, we plot a time-series showing the posterior expectation of the variable on each day. We also plot the model's posterior predictive probability that the agent would do the work on the last day (assuming the agent gets to the last day without having done the work). This feature is called `predictWorkLastMinute` in the codebox. (This feautre us a measure of whether the model predicts what actually happens on the last day). 


~~~~ 
// infer_procrastination

var observedStateAction = procrastinateUntilEnd102;
var lastChanceState = secondLast(observedStateAction)[0];


var posterior = function(observedStateAction, optimalModel) {
  var world = makeProcrastinationMDP2();
 
  return Enumerate(function(){
   
    var utilityTable = {reward: uniformDraw([0.5, 2, 3, 4, 5, 6, 7, 8]),
			waitCost: -0.1,
			workCost: -1};
    
    var params = {
      utility: makeProcrastinationUtility2(utilityTable),
      alpha: categorical([0.1, 0.2, 0.2, 0.2, 0.3], [0.1, 1, 10, 100, 1000]),
      discount: optimalModel ? 0 :  uniformDraw([0, .5, .1, 2, 4]),
      sophisticatedOrNaive: 'naive'
    };
    
    var agent = makeHyperbolicDiscounter(params, world);
    var act = agent.act;
    
    map(function(stateAction){
      var state = stateAction[0];
      var action = stateAction[1];
      factor( act(state, 0).score([], action) )
    }, observedStateAction);

    return {reward: utilityTable.reward, 
            alpha: params.alpha, 
            discount: params.discount, 
            predictWorkLastMinute: sample( act(lastChanceState, 0) ) == 'work'};
  });
};

var features = ['reward', 'predictWorkLastMinute', 'alpha', 'discount'];

// inference up to the t-th observation
var inferUpToTimeIndex = function(timeIndex){
  
  var expectations = function(erp){
    return map(function(feature){return expectation(getMarginal(erp,feature));
    }, features)
  };

  return map( function(optimal_or_hyperbolic){
    var observations = observedStateAction.slice(0,timeIndex);
    return expectations( posterior(observations, optimal_or_hyperbolic));
  }, [1,0]);
};

var indexToExpectations = map(inferUpToTimeIndex, range(observedStateAction.length));


// build full time series for each feature
var getTimeSeries = function(optimal_or_hyper){
  optimal_or_hyper == 0 ? print('Optimal Model:') : print('Possibly Discounting Model:');
  return map( function(i){
    var featureOut = map(function(optimal_hyper){return optimal_hyper[optimal_or_hyper][i];}, 
                         indexToExpectations);
    print('\n\n feature:' + features[i]); //, ' \n', featureOut);
    viz.line( range(observedStateAction.length), features[i] );
    return featureOut;
  }, range(features.length) );
};

print('Posterior expectation on feature after 
       observing agent "wait" for t timesteps (and "work" when t=9)');
map(getTimeSeries,[0,1]);
~~~~

When evaluating the two models, it's worth keeping in mind that the behavior we conditioned on is typical for humans. Suppose you hear someone has still not done a task with only two days left (where the cost for delaying is small and there's no risk of running out of time on the last day). Would you confidently rule out them doing it at the last minute? 

With two days left, the Optimal model has almost complete confidence that the agent doesn't care about the task enough to do the work (`reward < workCost`). Hence it assigns probability $$0.005$$ to the agent doing the task at the last minute (`predictWorkLastMinute`). By contrast, the Possibly Discounting model predicts the agent will do the task with probability around $$0.2$$. This probability is much higher because the model maintains the hypothesis that the agent values the reward enough to do it at the last minute (expectation for `reward` is 2.9 vs. 0.5). The probability is no higher than $$0.2$$ because the agent might be optimal (`discount==0`) or the agent might be too lazy to do the work even at the last minute (`discount` is high enough to overwhelm `reward`).

Suppose you now observe the person doing the task on the final day. What do you infer about them? The Optimal Model has to explain the action by massively revising its inference about `reward` and $$\alpha$$. It suddenly infers that the agent is extremely noisy and that `reward > workCost` by a big margin. The extreme noise is needed to explain why the agent would miss a good option nine out of ten times. By contrast, the Possibly Discounting Model does not change its inference about the agent's noise level very much at all (in terms of pratical significance). It infers a much higher value for `reward`, which is plausible in this context. [Point that Optimal Model predicts the agent will finish early on a similar problem, while Discounting Model will predict waiting till last minute.]


## Procrastination Example
HD causes big deviation in behavior. This is like smoker who smokes every day but wishes to quit.  Can you how inference gets stronger with passing days (online inference).

Probably talk about the structure of the procrastination MDP

~~~~
// non-discounter_completes_immediately

var world = makeProcrastinationMDP();

var utilityTable = {reward: 10,
		            procrastinationCost: -0.1,
					workCost: -1};
var utility = makeProcrastinationUtility(utilityTable);

var startState = {loc: "procrastinating",
		          procrastinationSteps: 0,
				  timeLeft: 10,
				  terminateAfterAction: false};

var agent = makeMDPAgent({utility: utility, alpha: 100}, world);
var trajectory = simulateMDP(startState, world, agent, 'stateAction');
map(function(stateAction){return [stateAction[0].loc, stateAction[1]];},
    trajectory);
~~~~

~~~~
// non-discounter_never_completes

var world = makeProcrastinationMDP();

var utilityTable = {reward: 0.8,
        		    procrastinationCost: -0.1,
					workCost: -1};
var utility = makeProcrastinationUtility(utilityTable);

var startState = {loc: "procrastinating",
		          procrastinationSteps: 0,
				  timeLeft: 5,
				  terminateAfterAction: false};

var agent = makeMDPAgent({utility: utility, alpha: 100}, world);
var trajectory = simulateMDP(startState, world, agent, 'stateAction');
map(function(stateAction){return [stateAction[0].loc, stateAction[1]];},
    trajectory);
~~~~


~~~~
// discounter_procrastinates

// procrastinates, because it thinks that it will do it later

var world = makeProcrastinationMDP();

var utilityTable = {reward: 10,
		            procrastinationCost: -0.1,
					workCost: -1};
var utility = makeProcrastinationUtility(utilityTable);

var startState = {loc: "procrastinating",
		          procrastinationSteps: 0,
				  timeLeft: 10,
				  terminateAfterAction: false};

var params = {utility: utility,
		      alpha: 100,
		      discount: 5,
		      sophisticatedOrNaive: 'naive'};
  
var agent = makeHyperbolicDiscounter(params, world);
var trajectory = simulateHyperbolic(startState, world, agent, 'stateAction');
map(function(stateAction){return [stateAction[0].loc, stateAction[1]];},
    trajectory);
~~~~

~~~~
// discounter_procrastination_varies_with_discount

// when the discount is small, the agent thinks that it should do the task
// immediately in order to avoid the procrastination cost, since it doesn't care
// about the workCost coming before the reward.
// when the discount is bigger, the agent procrastinates to the last moment, but
// eventually does the job because it's still better than nothing
// when the discount is very large, the agent always thinks that the utility of
// doing the work is below zero, so never does it.

var procrastinationWithDiscount = function(discount){
  var world = makeProcrastinationMDP();

  var utilityTable = {reward: 10,
		              procrastinationCost: -0.1,
		              workCost: -1};
  var utility = makeProcrastinationUtility(utilityTable);

  var startState = {loc: "procrastinating",
		            procrastinationSteps: 0,
		            timeLeft: 10,
		            terminateAfterAction: false};

  var params = {utility: utility,
		        alpha: 100,
		        discount: discount,
		        sophisticatedOrNaive: 'naive'};
  
  var agent = makeHyperbolicDiscounter(params, world);
  var trajectory = simulateHyperbolic(startState, world, agent, 'stateAction');
  return [last(trajectory)[1], trajectory.length];
};

var discounts = range(11);
var lastActionsAndTimes = map(procrastinationWithDiscount, discounts);

print('Discounts: ' + discounts + '\nLast actions and lengths of trajectories:'
      + JSON.stringify(lastActionsAndTimes));
~~~~


~~~~
// infer_procrastination

var posterior = function(stateActionPairs, allowDiscount) {
  return Enumerate(function(){
    var world = makeProcrastinationMDP();
  	var reward = uniformDraw([0.5, 2, 3, 4, 5, 6, 7, 8]);
	var alpha = categorical([0.1, 0.2, 0.2, 0.2, 0.3], [0.1, 1, 10, 100, 1000]);
	var discount = allowDiscount ? uniformDraw([0, .5, .1, 2, 4]) : 0;

    var utilityTable = {reward: reward,
			            procrastinationCost: -0.1,
						workCost: -1};
    var utility = makeProcrastinationUtility(utilityTable);
    var params = {utility: utility,
		          alpha: alpha,
		          discount: discount,
		          sophisticatedOrNaive: 'naive'};
    var agent = makeHyperbolicDiscounter(params, world);
    var act = agent.act;
    map(function(stateAction){
      factor(act(stateAction[0], 0).score([], stateAction[1]));
    }, stateActionPairs);

    return {reward: reward, alpha: alpha, discount: discount};
  });
};

// give it prefixes of the actual sequence, see how expected reward and logAlpha varies

var observedStateAction = procrastinateUntilEnd10;

var expectedFeaturesFromNStateActionPairs = function(n){
  var optimalPosterior = posterior(observedStateAction.slice(0,n), false);
  var hyperbolicPosterior = posterior(observedStateAction.slice(0,n), true);
  return {optimalReward: expectation(getMarginal(optimalPosterior,
						                          'reward')),
	      hyperbolicReward: expectation(getMarginal(hyperbolicPosterior,
		                                            'reward')),
          optimalAlpha: expectation(getMarginal(optimalPosterior,
						                         'alpha')),
	      hyperbolicAlpha: expectation(getMarginal(hyperbolicPosterior,
						                           'alpha')),
	      hyperbolicDiscount: expectation(getMarginal(hyperbolicPosterior,
						                              'discount'))};
};

var observedTimesteps = range(10);

var expectations = map(expectedFeaturesFromNStateActionPairs, observedTimesteps);
var expectedRewardsOptimal = map(function(object){return object.optimalReward;},
				                 expectations);
var expectedAlphasOptimal = map(function(object){return object.optimalAlpha;},
				                expectations);
var expectedRewardsHyperbolic = map(function(object){return object.hyperbolicReward;},
				                    expectations);
var expectedAlphasHyperbolic = map(function(object){return object.hyperbolicAlpha;},
				                   expectations);
var expectedDiscountsHyperbolic = map(function(object){return object.hyperbolicDiscount;},
                                      expectations);

print('Expected reward vs state action pairs observed, optimal');
viz.line(observedTimesteps, expectedRewardsOptimal);

print('Expected alpha vs state action pairs observed, optimal');
viz.line(observedTimesteps, expectedAlphasOptimal);

print('Expected reward vs state action pairs observed, hyperbolic');
viz.line(observedTimesteps, expectedRewardsHyperbolic);

print('Expected alpha vs state action pairs observed, hyperbolic');
viz.line(observedTimesteps, expectedAlphasHyperbolic);

print('Expected discount vs state action pairs observed, hyperbolic');
viz.line(observedTimesteps, expectedDiscountsHyperbolic);
~~~~


## Bandits

- Hyperbolic discounter and myopic-for-utility agent will explore less than optimal agent on both deterministic and stochastic bandits. We assume arm0 has a prize known to the agent and arm1 is uncertain to the agent (and we know the agent's prior). So the inference task is just to learn the utility the agent assigns to the prize from arm0. (We could assume that we know the utilities of the two possible prizes resulting from arm1). If the agent is discounting/myopic, they might take arm0, even if they don't have a very strong preference for the arm0 prize. As the time horizon gets longer, the difference in inference between the model that assumes optimality and the one that allows for discounting or myopia will get bigger. A graph illustrating this difference is in the NIPS paper. (With stochastic bandits you could have arms which are known to have high variance and with uncertain expectation. In this case you might get less exploration even if the myopia bound is higher or discounting is weaker. It'd be nice to include such an example but it's not neccesary).

~~~~
// infer_utility_from_no_exploration

var posterior = function(timeLeft, agentType) {
  var numArms = 2;
  var armToPrize = {0: 'chocolate',
		            1: 'nothing'};
  var worldAndStart = makeIRLBanditWorldAndStart(numArms, armToPrize, timeLeft);

  var startState = worldAndStart.startState;
  var alternativeLatent = update(armToPrize, {1: 'champagne'});
  var alternativeStartState = update(startState, {latentState:
						                          alternativeLatent});

  var priorAgentPrior = deltaERP(categoricalERP([0.7, 0.3],
						                        [startState,
												 alternativeStartState]));
  var priorPrizeToUtility = Enumerate(function(){
    var chocUtility = uniformDraw(range(20));
    return {chocolate: chocUtility,
	        nothing: 0,
			champagne: 20};
  });
  var prior = {priorAgentPrior: priorAgentPrior,
	           priorPrizeToUtility: priorPrizeToUtility};

  var baseAgentParams = {alpha: 100,
			             myopia: {on: agentType === 'myopic', bound: 1},
			             boundVOI: {on: false, bound: 0},
						 sophisticatedOrNaive: 'naive',
						 discount: agentType === 'hyperbolic' ? 1 : 0,
						 noDelays: agentType === 'optimal'};

  var stateAction = [[startState, 0]];

  var outputERP = inferIRLBandit(worldAndStart, baseAgentParams, prior,
				                 stateAction, 'offPolicy', 0, 'beliefDelay');

  return expectation(Enumerate(function(){
    return sample(outputERP).prizeToUtility.chocolate;
  }));
};

var timeLefts = range(10).slice(2);
var optimalExpectations = map(function(t){return posterior(t, 'optimal');},
			                   timeLefts);
var myopicExpectations = map(function(t){return posterior(t, 'myopic');},
			                 timeLefts);
var hyperbolicExpectations = map(function(t){return posterior(t, 'hyperbolic');},
				                 timeLefts);

print('Inference of utility of arm 0 for optimal agent as timeLeft increases');
viz.line(timeLefts, optimalExpectations);

print('Inference of utility of arm 0 for myopic agent as timeLeft increases');
viz.line(timeLefts, myopicExpectations);

print('Inference of utility of arm 0 for hyperbolic agent as timeLeft increases');
viz.line(timeLefts, hyperbolicExpectations);
~~~~



### Restaurant in Foreign City example
Model without myopia assumes a preference for the restaurants that are close (i.e. a prior that prefers them). If we add more restaurants, or make the variance higher and timecost lower, we can accentuate this effect. We can also make the game repeated (problem of tractability). 


