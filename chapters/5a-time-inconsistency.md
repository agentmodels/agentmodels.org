---
layout: chapter
title: "Time inconsistency"
description: Hyperbolic discounting, Naive and Sophisticated Agents, Formal Definition, Implementation

---

<!--
- small chapter on exponential discounting
- example or exercise with codebox for gridworld or bandits
-->

### Introduction
Time inconsistency is part of everyday human experience. In the night you wish to rise early; in the morning you prefer to sleep in. There is an inconsistency between what you prefer your future self to do and what your future self prefers to do. Forseeing this inconsistency, you take actions in the night to bind your morning self to get up. These range from setting an alarm clock to arranging for someone to drag you out of bed.

Similar examples abound. People pay upfront for gym subscriptions they rarely use. People procrastinate on writing papers: they plan to start the paper early but then delay until the last minute. The practical consequences of time inconsistency are substantial in different domains Cite:

- "Highbrow films gather dust: Time-inconsistent preferences and online DVD rentals"

- "Can Behavioral Tools Improve Online Student Outcomes? Experimental Evidence from a Massive Open Online Course" (patterson)

- experiment where state (AZ, NM?) offers free IUD and has big effect on teen pregnancy 

Time inconsistency has been used to explain not just quotidian laziness but also addiction, procrastination, impulsive behavior as well an array of "pre-commitment" behaviors refp:ainslie2001breakdown. Lab experiments of time inconsistency often use simple quantitative questions such as:

<blockquote><b>Question (1)</b>: Would you prefer to get $100 after 30 days or $110 after 31 days?
</blockquote>

Most people answer "yes" to Question (1). But a significant proportion of people reverse their earlier preference once the 30th day comes around and they contemplate getting $100 immediately. The next section describes a formal model of time preference that predicts this reversal. We incorporate into our model for MDP planning and implement it in WebPPL. 

### Time inconsistency due to hyperbolic discounting
Rational, utility-maximizing agents are often modeled as *discounting* future utilities/rewards relative to present rewards. Researchers in Machine Learning and Robotics construct systems for MDPs/RL with infinite time horizon that discount future rewards. Economists likewise model humans or firms discounting the infinite stream of future rewards. Justifications for discounting include (a) avoiding problems with expected utilities diverging and (b) capturing human preference for the near-term (e.g. due to interest rates, vague deadlines, the chance of not being around in the future to realize gains).

Discounting in these examples is *exponential*. An exponential discounting agent appears to have some kind of inconsistency over time. With a discount rate of 0.95 per day, $100 after 30 days is worth $21 and $110 at 31 days is $22. (assuming linear utility in money). Yet when the 30th day arrives, they are worth $100 and $105 respectively! (If instead the magnitudes were fixed from a starting time, the agent would have an overwhelming preference to travel back in time to get higher rewards!). Yet while these magnitudes have changed, the ratios stay fixed. Indeed, the ratios between any pair of outcomes are fixed regardless of the time the exponetial discounter evaluates them. So this agent thinks that two prospects in the far future are worth little compared to similar near-term prospects (disagreeing with his future self) but he agrees with his future self about which of the two future prospects is better. [TODO mention the relevance of this to planning in MDPs -- due to time consistency you only need compute a single expected utility for each state]. 

Any smooth discount function other than an exponential will result in preferences that reverse over time [cite]. So it's not so suprising that untutored humans should be subject to such reversals. (Without computational aids, human representations of numbers are systematically inaccurate. See refp:dehaene). Various functional forms for human discounting have been explored in the literature. We will describe the *hyperbolic discounting* model refp:ainslie2001breakdown because it is simple and well-studied. Any other functional form can easily be substituted into our models. 

Hyperbolic and exponential discounting curves are illustrated in Figure 1. We plot the discount factor $$D$$ as a function of time $$t$$ in days. The exponential is:

$$
D=\frac{1}{2^t}
$$

The hyperbolic function is:

$$
D=\frac{1}{1+2t}
$$

These are not realistic discount rates. The important difference is that the hyperbola is initially steep and then becomes almost flat, while the exponential continues to be steep. 

![Figure 1](/assets/img/hyperbolic_no_label.jpg). 

Consider the example above but with different numbers. You are offered $100 after 4 days or $110 after 5 days. The discount factors for 4 and 5 days from the present are labeled in Figure 2. The change in $$D$$ from day 4 to 5 is small for the hyperbola (so waiting for $110 is preferred) and big for the exponential. When Day 4 arrives, you can get $100 immediately or $110 after one day. The difference between the curves is labeled on the left. The hyperbola is now steep and leads to you taking the $100 -- reversing your earlier preference.

**Exercise**: Calculate the discounted utilities for the two options ($100 vs. $110) for both hyperbolic and exponential discounting. First compute them when the $100 is 4 days from the present, then when it's 3 days from the present and so on (up to when it's 0 days from the present). 

![Figure 2](/assets/img/hyperbolic_label.jpg). 


### Time inconsistency and sequential decision problems
We have shown that hyperbolic discounters have different preferences over the $100 and $110 depending on when they make the evaluation. This conflict in preferences leads to complexities in planning that don't occur in the optimal, non-discounting (PO)MDP agents from previous chapters (or in exponential discounters in infinite horizon problems).

Imagine you are in the situation of Question (1) and have the time inconsistent preferences. You get to write down your preference but after 30 days you'll be free to change your mind. If you know your future self will choose the $100 immediately, you will pay a small cost now to pre-commit your future self. (Maybe you re-schedule an important meeting to 30 from now so you can't go and get the money). However, if you believe your future self will share your preferences, you won't pay this cost (and so you'll end up taking the $100). This illustrates a key distinction between time inconsistent agents solving sequential problems:

- **Naive agent**: assumes his future self shares his current time preference exactly. So a Naive hyperbolic discounter assumes his far future self has a nearly flat discount curve (when in reality his future self has "steep then flat" discount curve). 

- **Sophisticated agent**: has the correct model of his future self's time preference. So a Sophisticated hyperbolic discounter has a nearly flat discount curve for the far future but is aware that his future self does not share this discount curve. 

The Naive agent chooses actions based on the false assumption that his future selves share his time preference. POMDP agents are *uncertain* about some features of the environment but this uncertainty can be corrected. Naive have a fundamentally wrong model of the environment (due to an inaccurate model of themselves) that they don't correct by observation.

Sophisticated agents have an accurate model of their future selves. This enables a Sophisticated agent, acting at time $$t_0$$, to pre-commit his future self at times $$t>t_0$$, to actions that the $$t_0$$-agent prefers. So if pre-commitment actions are available at time $$t_0$$, we expect the $$t_0$$-agent to do better (by its own $$t_0$$ lights) if it's Sophisticated rather than Naive -- since if Sophisticated it has identical preferences and more knowledge of the world. This means that being Naive at $$t_0$$ is better for the preferences of the $$t>t_0$$ agents.


### Naive and Sophisticated Agents: Gridworld Example
Before describing our formal model and implementation of Naive and Sophisticated hyperbolic discounters, we illustrate their contrasting behavior using the Restaurant Choice example. We use the MDP version, where the agent has full knowledge of the locations of restaurants and of which restaurants are open. Recall the problem setup: 

<blockquote>
Bob is looking for a place to eat. His decision problem is to take a sequence of actions such that (a) he eats at a restaurant he likes and (b) he does not spend too much time walking. The restaurant options are: the Donut Store, the Vegetarian Salad Bar, and the Noodle Shop. The Donut Store is a chain with two local branches. We assume each branch has identical utility for Bob. We abbreviate the restaurant names as "Donut South", "Donut North", "Veg" and "Noodle".
</blockquote>

The only difference from previous versions of Restaurant Choice is that we model the restaurants as providing *two* utilities. The agent first receives the *immediate reward* (e.g. how good the food tastes) and then (at some fixed time delay) receives the *delayed reward* (e.g. how good the person feels after eating it). Here is the code that uses the Gridworld library to construct the MDP.

**Exercise:** Before scrolling down, predict how Naive and Sophisticated hyperbolic discounters with identical preferences could differ in their actions on this problem.

[codebox: bigGridworld. draw with agent starting in 3,1.]

We now consider two hyperbolic discounting agents with the same preferences and discounting function but where one is Naive and the other Sophisticated.

[codeboxes with both Naive and Soph. Or one codebox with both and some parameter to control Naive/Soph easily.]

**Exercise:** Before reading further, your goal is to do preference inference from the observed actions in the codebox above (using only a pen and paper). The discount function is the hyperbola $$D=\frac{1}{1+kt}$$, where $$t$$ is the time from the present, $$D$$ is the discount factor (multiplied by the utility) and $$k$$ is a positive constant. Work out a full set of parameters for the agent that predict the observed behavior. This includes utilities for the restaurants (both *immediate* and *delayed*) and for the `timeCost`, as well as the discount constant $$k$$. (Assume there is no softmax noise). 

The Naive agent goes to Donut North, even though Donut South (which has identical utility) is closer to the agent's starting point. One explanation is that the Naive agent prefers Veg (ignoring discounting). At the start, no restaurants can be reached quickly and so the agent's discount factor is nearly flat when evaluating each one of them. This makes Veg look most attractive. But going to Veg means getting closer to Donut North, which becomes more attractive than Veg once the agent is close to it. (Taking an inefficient path -- one that is dominated by another path -- is typical of time inconsistent agents). 

The Sophisticated agent, when considering its actions from the starting point, can accurately model what it *would* do if it ended up adjacent to Donut North. So it avoids temptation by taking the long, inefficient route to Veg. 

In this simple example, the Naive and Sophisticated agents each take paths that optimal time-consistent MDP agents never take. While a time-consistent agent with high softmax noise would take the Naive agent's path with low probability, the Sophisticated path has massively lower probability for such an agent. So this is an example where a bias leads to a *systematic* deviation from optimality and behavior that is not predicted by an optimal model. In a later chapter we explore inference of inferences for time inconsistent agents.

### Formal Model of Naive and Sophisticated Hyperbolic Discounters

To formalize Naive and Sophisticated hyperbolic discounting, we make a small modificiation to the MDP agent model. The key idea is to add an additional variable for measuring time, the *delay*, which is distinct from the objective time index (called `timeLeft` our implementation). Although the environment is stationary, the objective time remaining is important in planning for finite-horizon MDPs because it determines how far the agent can travel or explore before time is up. The delays are *subjective*: they are used by the agent in *evaluating* possible future rewards but they are not an independent feature of the decision problem.

We use delays because discounting agents have time preference. When evaluating future rewards, they need to keep track of how far ahead in time that reward occurs, i.e. keep track of the time-delay in getting the reward. Naive and Sophisticated agents evaluate future rewards in the same way. They differ in how they simulate their future actions.

The Naive agent at objective time $$t$$ assumes his future self at objective time $$t+c$$ (where $$c>0$$) shares his time preference. So he simulates the $$(t+c)$$-agent as evaluating a reward at time $$t+c$$ with delay $$d=c$$ (hence discount factor $$\frac{1}{1+kc}$$) rather than the true delay $$d=0$$. The Sophisticated agent correctly models his $$(t+c)$$-agent future self as evaluating an immediate reward with delay $$d=0$$ and hence a discount factor of one (i.e. no discounting). 

Adding delays to our model is straightforward. In defining the MDP agent, we presented Bellman-style recursions for the expected utility of state-action pairs. Discounting agents evaluate states and actions differently depending on their *delay* from the present. So we now define expected utilities of state-action-delay triples:

$$
EU[s,a,d] = \delta(d)U(s, a) + E_{s', a'}(EU[s', a',d+1])
$$

where:

- $$\delta  \colon \mathbb{N} \to \mathbb{R}$$ is the discount function from the delay to the discount factor. In our examples we have (where $$k>0$$ is the discount constant):

$$
\delta(d) = \frac{1}{1+kd}
$$

- $$s' \sim T(s,a)$$ exactly as in the non-discounting case.

- $$a' \sim C(s'; d_P)$$ where $$d_P=0$$ for Sophisticated and $$d_P=d+1$$ for Naive.


The function $$C \colon S \times \mathbb{N} \to A$$ is again the *act* function. For $$C(s'; d+1)$$ we take a softmax over the expected value of each action $$a$$, namely, $$EU[s',a,d+1]$$. The act function now takes a delay argument. We interpret $$C(s';d+1)$$ as "the softmax action the agent would take in state $$s'$$ given that their rewards occur with a delay $$d+1$$".

The Naive agent simulates his future actions by computing $$C(s';d+1)$$; the Sophisticated agent computes the action that will *actually* occur, which is $$C(s';0)$$. So if we want to simulate an environment including a hyperbolic discounter, we can compute the agent's action with $$C(s;0)$$ for every state $$s$$. 


### Implementing the hyperbolic discounter
As with the MDP and POMDP agents, our WebPPL implementation directly translates the mathematical formulation of Naive and Sophisticated hyperbolic discounting. The variable names correspond as follows:

- The function $$\delta$$ is named `discountFunction`

- The "perceived delay", which is the delay from which the agent's simulate future self evaluates rewards, is $$d_P$$ in the math and `perceivedDelay` below. 

- $$s'$$, $$a'$$, $$d+1$$ correspond to `nextState`, `nextAction` and `delay+1` respectively. 

[TODO: add John's changing expected utilities, with an explanation of them]

<!--code from scratch/agentModelsHyperbolic.wppl]-->
~~~~


var makeAgent = function (params, world) {
  var stateToActions = world.stateToActions;
  var transition = world.transition;
  var utility = params.utility;

  var discountFunction = function(delay){
    return 1/(1 + params.discount*delay);
  };

  var isNaive = params.sophisticatedOrNaive=='naive';
    
  var act = dp.cache( 
    function(state, delay){
      return Enumerate(function(){
        var action = uniformDraw(stateToActions(state));
        var eu = expectedUtility(state, action, delay);    
        factor(params.alpha * eu);
        return action;
      });      
    });
  
  var expectedUtility = dp.cache(
    function(state, action, delay){
      var u = discountFunction(delay) * utility(state, action);
      if (state.terminateAfterAction){
        return u; 
      } else {                     
        return u + expectation( Enumerate(function(){
          var nextState = transition(state, action); 
          var perceivedDelay = isNaive ? delay + 1 : 0;
          var nextAction = sample(act(nextState, perceivedDelay));
          return expectedUtility(nextState, nextAction, delay+1);  
        }));
      }                      
    });
  
  return {
    params : params,
    expectedUtility : expectedUtility,
    act: act
  };
};

~~~~


Next we simulate both the naive and sophisticated versions of our hyperbolic discounter. 

To better understand the behavior of the discounter, we use the `plannedTrajectories` function to compute the agent's current plan at each timestep. `plannedTrajectories` computes what full path the agent currently expects its future self to take. The naive agent, the plan is systematically wrong; the naive agent thinks in the future it will value rewards the same way it does now, but in reality it will discount them differently. The sophisticated agent on the other hand, correctly anticipates its future actions, the agent knows that in the future it will value rewards differently that it does now. 

We can animate these expected paths by passing the optional `paths` argument to `GridWorld.draw`.

Watch the simulation and notice how the naive agent changes its plan to go to Veg as it passes by Donut N. It failed to anticipate that it would be sidetracked by Donut N. The sophisticated agent, on the other hand anticipates this and routes around Donut N. 

~~~~
var makeAgent = function (params, world) {
  var stateToActions = world.stateToActions;
  var transition = world.transition;
  var utility = params.utility;

  var discountFunction = function(delay){
    return 1/(1 + params.discount*delay);
  };

  var isNaive = params.sophisticatedOrNaive=='naive';
    
  var act = dp.cache( 
    function(state, delay){
      return Enumerate(function(){
        var action = uniformDraw(stateToActions(state));
        var eu = expectedUtility(state, action, delay);    
        factor(params.alpha * eu);
        return action;
      });      
    });
  
  var expectedUtility = dp.cache(
    function(state, action, delay){
      var u = discountFunction(delay) * utility(state, action);
      if (state.terminateAfterAction){
        return u; 
      } else {                     
        return u + expectation( Enumerate(function(){
          var nextState = transition(state, action); 
          var perceivedDelay = isNaive ? delay + 1 : 0;
          var nextAction = sample(act(nextState, perceivedDelay));
          return expectedUtility(nextState, nextAction, delay+1);  
        }));
      }                      
    });
  
  return {
    params : params,
    expectedUtility : expectedUtility,
    act: act
  };
};

var simulate = function(startState, world, agent) {
  var act = agent.act;
  var expectedUtility = agent.expectedUtility;
  var transition = world.transition;

  var sampleSequence = function (state) {
    var delay = 0;
    var action = sample(act(state, delay));
    var nextState = transition(state, action); 
    var out = [state,action]
    return state.terminateAfterAction ?
      [out] : [out].concat(sampleSequence(nextState));
  };
  return sampleSequence(startState);
};


// Construct MDP, i.e. world
var startState = { 
  loc : [3,0],
  terminateAfterAction : false,
  timeLeft : 13
};

var world = makeDonutWorld2({ big : true, maxTimeAtRestaurant : 2});


// Construct hyperbolic discounting agent


// Utilities for restaurants: [immediate reward, delayed reward]
// Also *timeCost*, cost of taking a single action.

var restaurantUtility = makeRestaurantUtilityFunction(world, {
    'Donut N' : [10, -10],
    'Donut S' : [10, -10],
    'Veg'   : [-10, 20],
    'Noodle': [0, 0],
    'timeCost': -.01
});

var baseAgentParams = {
  utility : restaurantUtility,
  alpha : 500, 
  discount : 1
};

// Construct Sophisticated and Naive agents
var sophisticatedAgent = makeAgent(
  update(baseAgentParams, {sophisticatedOrNaive: 'sophisticated'}), 
  world
);

var trajectory = simulate(startState, world, sophisticatedAgent); 
var plans = plannedTrajectories(trajectory, world, sophisticatedAgent);
Gridworld.draw(world, { trajectory : trajectory, paths : plans });

var naiveAgent = makeAgent( 
  update(baseAgentParams, {sophisticatedOrNaive: 'naive'}), 
  world
);

var trajectory = simulate(startState, world, naiveAgent); 
var plans = plannedTrajectories(trajectory, world, naiveAgent);
Gridworld.draw(world, { trajectory : trajectory, paths : plans });
~~~~
