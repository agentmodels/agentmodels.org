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


## Procrastination Example
HD causes big deviation in behavior. This is like smoker who smokes every day but wishes to quit.  Can you how inference gets stronger with passing days (online inference).


## Bandits

- Hyperbolic discounter and myopic-for-utility agent will explore less than optimal agent on both deterministic and stochastic bandits. We assume arm0 has a prize known to the agent and arm0 is uncertain to the agent (and we know the agent's prior). So the inference task is just to learn the utility the agent assigns to the prize from arm0. (We could assume that we know the utilities of the two possible prizes resulting from arm1). If the agent is discounting/myopic, they might take arm0, even if they don't have a very strong preference for the arm0 prize. As the time horizon gets longer, the difference in inference between the model that assumes optimality and the one that allows for discounting or myopia will get bigger. (With stochastic bandits you could have arms which are known to have high variance and with uncertain expectation. In this case you might get less exploration even if the myopia bound is higher or discounting is weaker. It'd be nice to include such an example but it's not neccesary). 



### Restaurant in Foreign City example
Model without myopia assumes a preference for the restaurants that are close (i.e. a prior that prefers them). If we add more restaurants, or make the variance higher and timecost lower, we can accentuate this effect. We can also make the game repeated (problem of tractability). 


