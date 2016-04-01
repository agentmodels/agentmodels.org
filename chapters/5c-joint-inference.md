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

We return to the Restaurant Choice example and consider inference for the MDP and POMDP versions:

1. In the MDP setting (where the agent has full knowledge), we compare a model that assumes an optimal, non-discounting MDP agent to a model that allows for time-inconsistency (but also includes a non-discounting optimal agent in the hypothesis space).

2. In the POMDP setting (where the restaurants may be open or closed and the agent can learn this from observation), we do joint inference over preferences, beliefs and discounting behavior. We show that our inference approach can produce multiple explanations for the same behavior and that explanations in terms of beliefs and preferences are more plausible than those involving time-inconsistency. 

As we discussed in Chapter V.1, time-inconsistent agents can produce trajectories on the MDP (full knowledge) version of this scenario that never occur for an optimal agent without noise. 

In our first inference example, we do joint inference over preferences, softmax noise and the discounting behavior of the agent. (We assume for this example that the agent has full knowledge and is not Myopic). We compare the preference inferences [that allow for possibility of time inconsistency] to the earlier inference approach that assumes optimality.

### Example 1: Time-inconsistent vs. optimal MDP agents
This example compares a model that assumes an optimal agent (and just infers their preferences and softmax noise) to a model that also allows for sub-optimal time-inconsistent agents. Before making a direct comparison, we demonstrate that we can infer the preferences of time-inconsistent agents from observations of their behavior.

#### Assume discounting, infer "Naive" or "Sophisticated"
First we condition on the agent moving to Donut North, which is distinctive to the Naive hyperbolic discounter:

~~~~
var world = makeRestaurantChoiceMDP();
var path = map(first,restaurantNameToObservationTime11['naive']);         
GridWorld.draw(world, {trajectory:path});
~~~~

For inference, we specialize the approach above for (possibly time-inconsistent) agents in MDPs. So we infer $$\nu$$ and $$k$$ (the hyperbolic discounting parameters) but not the initial belief state $$b_0$$. The function `exampleGetPosterior` is a slightly simplified version of the library function we use below.

~~~~
var exampleGetPosterior = function(world, priorUtilityTable, priorDiscounting,
    priorAlpha, observedStateAction){
  return Enumerate(function () {

    // Sample parameters from prior
    var utilityTable = priorUtilityTable();
    var sophisticatedOrNaive = priorDiscounting().sophisticatedOrNaive;

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

    // return parameters
    var vegMinusDonut = sum(utilityTable['Veg']) - sum(utilityTable['Donut N']);
    return {
      utility: utilityTable, 
      sophisticatedOrNaive: discounting.sophisticatedOrNaive,
      vegMinusDonut: vegMinusDonut,
    };
  });
};
exampleGetPosterior;  
~~~~

This inference function allows for inference over the softmax `alpha` parameter and the discount constant `discount`. For this example, we fix these values so that the agent has low noise and `discount==1`. We also fix the `timeCost` utility to be small and negative, as well as the utility of Noodle to be negative. So we infer only the agent's preferences and whether they are Naive or Sophisticated.

~~~~
// Call to hyperbolic library function and helper display function
///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(erp){
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg']);
  print('... and for Donut: ' + utility['Donut N'] + ' \n')
  viz.vegaPrint(getMarginalObject(erp,'vegMinusDonut'));
  viz.vegaPrint(getMarginalObject(erp,'sophisticatedOrNaive'));
};
///

// Prior on agent's utility function
var priorUtilityTable = function(){
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
var prior = {utilityTable:priorUtilityTable, discounting:priorDiscounting, alpha:priorAlpha};

// Get world and observations
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['naive'];
var posterior = getPosterior(world, prior, observedStateAction);
displayResults(posterior);
~~~~

The graphs display the posterior after conditioning on the behavior depicted above. The variable `vegMinusDonut` is the difference in total utility between Veg and Donut. Inference rules out cases where the total utility is equal, since the agent would simply go to Donut South in that case. As expected, we infer that the agent is Naive.

Using the same prior, we condition on the path distinctive to the Sophisticated agent:

~~~~
var world = makeRestaurantChoiceMDP();
var path = map(first, restaurantNameToObservationTime11['sophisticated']);         
GridWorld.draw(world, {trajectory:path});
~~~~

Here are the results of inference: 

~~~~
// Definition of world, prior and inference function is same as above codebox

///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(erp){
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg']);
  print('... and for Donut: ' + utility['Donut N'] + ' \n')
  viz.vegaPrint(getMarginalObject(erp,'vegMinusDonut'));
  viz.vegaPrint(getMarginalObject(erp,'sophisticatedOrNaive'));
};

// Prior on agent's utility function
var priorUtilityTable = function(){
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
var prior = {utilityTable:priorUtilityTable, discounting:priorDiscounting, alpha:priorAlpha};
///

// Get world and observations
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['sophisticated'];
var posterior = getPosterior(world, prior, observedStateAction);
displayResults(posterior);
~~~~

If the agent goes directly to Veg, then they don't provide information about whether they are Naive or Sophisticated. Using the same prior again, we do inference on this path:

~~~~
var world = makeRestaurantChoiceMDP();
var path = map(first, restaurantNameToObservationTime11['vegDirect']);         
GridWorld.draw(world, {trajectory:path});
~~~~

Here are the results of inference: 

~~~~
// Definition of world, prior and inference function is same as above codebox

///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(erp){
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg']);
  print('... and for Donut: ' + utility['Donut N'] + ' \n')
  viz.vegaPrint(getMarginalObject(erp,'vegMinusDonut'));
  viz.vegaPrint(getMarginalObject(erp,'sophisticatedOrNaive'));
};

// Prior on agent's utility function
var priorUtilityTable = function(){
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
var prior = {utilityTable:priorUtilityTable, discounting:priorDiscounting, alpha:priorAlpha};
///

// Get world and observations
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['vegDirect'];
var posterior = getPosterior(world, prior, observedStateAction);
displayResults(posterior);
~~~~

#### Assume non-discounting, infer preferences and softmax
We want to compare a model that assumes an optimal MDP agent with one that allows for time-inconsistency. We first show the inferences by the model that assumes optimality. This model has to explain the paths distinctive of Naive and Sophisticated agents in terms of softmax noise.

~~~~
// Same helper functions as above
///fold:
var restaurantHyperbolicInfer = getRestaurantHyperbolicInfer();
var getPosterior = restaurantHyperbolicInfer.getPosterior;

var displayResults = function(erp){
  var utility = erp.MAP().val.utility;
  print('MAP utility for Veg: ' + utility['Veg']);
  print('... and for Donut: ' + utility['Donut N'] + ' \n')
  viz.vegaPrint(getMarginalObject(erp,'vegMinusDonut'));
  viz.vegaPrint(getMarginalObject(erp,'sophisticatedOrNaive'));
  viz.vegaPrint(getMarginalObject(erp,'alpha'));
};
///

// Prior on agent's utility function
var priorUtilityTable = function(){
  var utilityValues = [-10,0,10,20,30,40];
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
var prior = {utilityTable:priorUtilityTable, discounting:priorDiscounting, alpha:priorAlpha};

// Get world and observations
var world = makeRestaurantChoiceMDP({noReverse:false});
var observedStateAction = restaurantNameToObservationTime11['naive'];
var posterior = getPosterior(world, prior, observedStateAction);
displayResults(posterior);
~~~~






### Naive/Soph/Neutral examples for Restaurant Choice Gridworld

- Recover correct inference about preferences and agent type in three simple cases (no beliefs)

- Two or three scenario example. Softmaxing is ruled out in favor of naive/soph

- Joint inference for Naive and Soph examples:

In Naive, we consider false belief about Donut South being closed, preference for Donut North over south (unlikely in the prior) and the discounting explanation. Show multimodal inference. Discuss identification issues and Bayes Occam.

For Soph, we compare false belief the Noodle is open, a positive timeCost (which is has a low prior probability), and the Soph explanation. We could mention the experiment showing that when belief / preference explanations were possible, people tended to prefer them over HD explanations (though they did generate HD explanations spontaneously).

- Big inference example (maybe in later chapter): use HMC and do inference oiver cts params. 

### Procrastination Example
HD causes big deviation in behavior. This is like smoker who smokes every day but wishes to quit.  Can you how inference gets stronger with passing days (online inference).


### Bandits

- Hyperbolic discounter and myopic-for-utility agent will explore less than optimal agent on both deterministic and stochastic bandits. We assume arm0 has a prize known to the agent and arm0 is uncertain to the agent (and we know the agent's prior). So the inference task is just to learn the utility the agent assigns to the prize from arm0. (We could assume that we know the utilities of the two possible prizes resulting from arm1). If the agent is discounting/myopic, they might take arm0, even if they don't have a very strong preference for the arm0 prize. As the time horizon gets longer, the difference in inference between the model that assumes optimality and the one that allows for discounting or myopia will get bigger. (With stochastic bandits you could have arms which are known to have high variance and with uncertain expectation. In this case you might get less exploration even if the myopia bound is higher or discounting is weaker. It'd be nice to include such an example but it's not neccesary). 



### Restaurant in Foreign City example
Model without myopia assumes a preference for the restaurants that are close (i.e. a prior that prefers them). If we add more restaurants, or make the variance higher and timecost lower, we can accentuate this effect. We can also make the game repeated (problem of tractability). 


