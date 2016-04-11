---
layout: chapter
title: "Time inconsistency"
description: Hyperbolic discounting, Naive and Sophisticated Agents, Formal Definition, Implementation

---


### Introduction
Time inconsistency is part of everyday human experience. In the night you wish to rise early; in the morning you prefer to sleep in. There is an inconsistency between what you prefer your future self to do and what your future self prefers to do. Forseeing this inconsistency, you take actions in the night to bind your morning self to get up. These range from setting an alarm clock to having someone drag you out of bed.

Similar examples abound. People pay upfront for gym subscriptions they rarely use. People procrastinate on writing papers: they plan to start the paper early but then delay until the last minute. The practical consequences of time inconsistency are substantial in different domains Cite: ("Highbrow films gather dust: Time-inconsistent preferences and online DVD rentals"), ("Can Behavioral Tools Improve Online Student Outcomes? Experimental Evidence from a Massive Open Online Course" (patterson)). Time inconsistency has been used to explain not just quotidian laziness but also addiction, procrastination, impulsive behavior as well an array of "pre-commitment" behaviors refp:ainslie2001breakdown. Lab experiments of time inconsistency often use simple quantitative questions such as:

>**Question**: Would you prefer to get $100 after 30 days or $110 after 31 days?

Most people prefer the $110. But a significant proportion of people reverse their earlier preference once the 30th day comes around and they contemplate getting $100 immediately. This chapter describes a formal model of time inconsistency that predicts this reversal. We incorporate this model into our MDP agent and implement it in WebPPL.

### Time inconsistency due to hyperbolic discounting

This chapter explores the model time inconsistency as resulting from *hyperbolic discounting*. The idea is that humans prefer receiving the same rewards sooner rather than later and the *discount function* describing this quantitatively is a hyperbola. Before describing the hyperbolic model, we provide some background on time discounting and show how it can easily be added to the agent models of the previous chapter.

#### Exponential discounting for optimal agents

The examples of decision problems in previous chapters have a *known*, *finite* time horizon. Yet there are practical decision problems that are better modeled as having an *unbounded* or *infinite* time horizon. In Machine Learning, many RL problems have an uncertain time horizon (e.g. play a video game, drive a car from A to B). Human economic decisions have a time horizon that is both long (e.g. 50 years) and uncertain, and economists often assume unbounded time.

Generalizing the agent model from previous chapters to the unbounded case faces a difficulty. The *infinite* summed expected utility of an action will (for most natural sequential decision problems) not converge. The standard solution is to model the agent as maximizing the *discounted* expected utility, where the discount function is exponential and has a single free parameter. This makes the infinite sums converge and results in an agent model that is analytically and computationally tractable. 

Aside from mathematical convenience, there is an additional justification for exponential discounting in models of rationl agents. A model of rational agents should not limit the kinds of things or properties the agents can care about[^justification]. In particular, such a model should permit a generalized preference for desirable things occurring sooner rather than later. Exponential discounting models such a preference and makes the agent time consistent[^exponential].

[^justification]: Rational agents can't have inconsistent preferences but aside from this a model should not constrain the kinds of things they care about. More concretely, people care about a range of things: e.g. the food they eat daily, their careers, their families, the progress of science, the preservation of the earth's environment. Many have argued that humans have a time preference. So models that learn human preferences should allow for this possibility. 

[^exponential]: There are arguments that exponential discounting is the uniquely rational mode of discounting for agents with time preference. See X and Y for critical discussion (toby, dasgupta, wolfgang schwarz). 

It is straightforward to add exponential discounting to our existing agent models. We explain this in detail below. Before that we illustrate the effects of exponential discounting. We return to the deterministic Bandit problem from Chapter III.3. Suppose a person decides every year where to go on vacation. There is a fixed list of options and a finite time horizon[^bandit]. The person discounts exponentially and so they prefer a good vacation now to an even better one in the future. This means they are less likely to *explore*, since exploration takes time to pay off.

[^bandit]: As noted above, exponential discounting is usually combined with an *unbounded* time horizon. However, if a human makes a series of decisions over a long time scale, then it makes sense to include their time preference. For this particular example, imagine the person is looking for the best skiing or sports facilities and doesn't care about variety. There could be a known finite time horizon because they won't anymore be able to take a long vacation or they are too old for the sport. 

TODO: Show deterministic bandits with say 5 arms. agent is certain about some but many are uncertain. Agent will explore the one's with fairly high EV but not the ones with high variance.[Alternatively, could do a gridworld example and show the agent who prefers veg goes to donuth south. Or could do some simple numerical computations for bandits.]


 
#### Discounting and time inconsistency
Exponential discounting is typically thought of as a *relative* time preference. A fixed reward will be discounted by a factor of $$\delta^{-30}$$ if received on Day 30 rather than Day 0. On Day 30, the same reward is discounted by $$\delta^{-30}$$ if received on Day 60 and not at all if received on Day 30. This is how exponential discounting is implemented in Reinforcement Learning. This relative time preference is "inconsistent" in a certain superficial sense. With $$\delta=0.95$$ per day (and linear utility in money), $100 after 30 days is worth $21 and $110 at 31 days is worth $22. Yet when the 30th day arrives, they are worth $100 and $105 respectively[^inconsistent]! Yet while these magnitudes have changed, the ratios stay fixed. Indeed, the ratios between any pair of outcomes are fixed regardless of the time the exponetial discounter evaluates them. So this agent thinks that two prospects in the far future are worth little compared to similar near-term prospects (disagreeing with his future self) but he agrees with his future self about which of the two future prospects is better.

[^inconsistent]: One can think of exponential discounting in a non-relative way by choosing a fixed staring time in the past (e.g. the agent's birth) and discounting everything relative to that. This results in an agent with a preference to travel back in time to get higher rewards!

Any smooth discount function other than an exponential will result in preferences that reverse over time [TODO: cite a good statement and explanation of this result]. So it's not so suprising that untutored humans should be subject to such reversals[^reversal]. Various functional forms for human discounting have been explored in the literature. We will describe the *hyperbolic discounting* model refp:ainslie2001breakdown because it is simple and well-studied. Any other functional form can easily be substituted into our models.

[^reversal]: Without computational aids, human representations of discrete and continuous quantities (including durations in time and dollar values) are systematically inaccurate. See refp:dehaene. 

Hyperbolic and exponential discounting curves are illustrated in Figure 1. We plot the discount factor $$D$$ as a function of time $$t$$ in days, with constants $$\delta$$ and $$k$$ controlling the slope of the function. In this example, each constant is set to 2. The exponential is:

$$
D=\frac{1}{\delta^t}
$$

The hyperbolic function is:

$$
D=\frac{1}{1+kt}
$$

The crucial difference between the curves is that the hyperbola is initially steep and then becomes almost flat, while the exponential continues to be steep. This means that exponential discounting is time consistent and hyperbolic discounting is not. 

TODO: add Figure 1 label, put the function forms 1/2^t and 1/(1+2t) in the legend. maybe label the parts that are steep / shallow. 

![Figure 1](/assets/img/hyperbolic_no_label.jpg)

>**Exercise:** We return to our running example but with slightly different numbers. The agent chooses between receiving $100 after 4 days or $110 after 5 days. The goal is to compute the preferences over each option for both exponential and hyperbolic discounters, using the discount curves shown in Figure 1. Compute the following:

> 1. The discounted utility of the $100 and $110 rewards relative to Day 0 (i.e. how much the agent values each option when the rewards are 4 or 5 days away).
>2. The discounted utility of the $100 and $110 rewards relative to Day 4 (i.e. how much each option is valued when the rewards are 0 or 1 day away).

### Time inconsistency and sequential decision problems
We have shown that hyperbolic discounters have different preferences over the $100 and $110 depending on when they make the evaluation. This conflict in preferences leads to complexities in planning that don't occur in the optimal (PO)MDP agents which either discount exponentially or do not discount at all.

Returning to the previous example (see exercise above), imagine you have time inconsistent preferences. On Day 0, you write down your preference but on Day 4 you'll be free to change your mind. If you know your future self would choose the $100 immediately, you'd pay a small cost now to *pre-commit* your future self. However, if you believe your future self will share your current preferences, you won't pay this cost (and so you'll end up taking the $100). This illustrates a key distinction between Naive and Sophisticated agents:

- **Naive agent**: assumes his future self shares his current time preference. For example, a Naive hyperbolic discounter assumes his far future self has a nearly flat discount curve (rather than the "steep then flat" discount curve he actually has). 

- **Sophisticated agent**: has the correct model of his future self's time preference. A Sophisticated hyperbolic discounter has a nearly flat discount curve for the far future but is aware that his future self does not share this discount curve.

Both kinds of agents will value rewards differently at different times. To distinguish a hyperbolic discounter's current and future selves, we refer to the agent acting at time $$t_i$$ as the $$t_i$$-agent. A Sophisticated agent, unlike a Naive agent, has an accurate model of his future selves. The Sophisticated $$t_0$$-agent predicts the actions of the $$t$$-agents (for $$t>t_0$$) that would conflict with his preferences. To prevent these actions, the $$t_0$$-agent tries to take actions that *pre-commit* the future agents to outcomes the $$t_0$$-agent prefers[^sophisticated].

[^sophisticated]: As has been pointed out previously, there is a kind of "inter-generational" conflict between agent's future selves. If pre-commitment actions are available at time $$t_0$$, the $$t_0$$-agent does better in expectation if it is Sophisticated rather than Naive. Equivalently, the $$t_0$$-agent's future selves will do better if the agent is Naive.


### Naive and Sophisticated Agents: Gridworld Example
Before describing our formal model and implementation of Naive and Sophisticated hyperbolic discounters, we illustrate their contrasting behavior using the Restaurant Choice example. We use the MDP version, where the agent has full knowledge of the locations of restaurants and of which restaurants are open. Recall the problem setup: 

>Bob is looking for a place to eat. His decision problem is to take a sequence of actions such that (a) he eats at a restaurant he likes and (b) he does not spend too much time walking. The restaurant options are: the Donut Store, the Vegetarian Salad Bar, and the Noodle Shop. The Donut Store is a chain with two local branches. We assume each branch has identical utility for Bob. We abbreviate the restaurant names as "Donut South", "Donut North", "Veg" and "Noodle".

The only difference from previous versions of Restaurant Choice is that restaurants now have *two* utilities. On entering a restaurant, the agent first receives the *immediate reward* (i.e. how good the food tastes) and at the next timestep receives the *delayed reward* (i.e. how good the person feels after eating it).

**Exercise:** Before scrolling down, predict how Naive and Sophisticated hyperbolic discounters with identical preferences could differ for the Restaurant Choice problem shown in the codebox immediately below.

~~~~
// draw_choice
var world = makeRestaurantChoiceMDP();
var startState = restaurantChoiceStart;
print('starting state is: ' + JSON.stringify(startState) );
GridWorld.draw(world, {trajectory:[startState]});
~~~~

The next two codeboxes show the behavior of two hyperbolic discounters. Each agent has the same preferences and discount function. They differ only in that the first is Naive and the second is Sophisticated.

~~~~
// draw_naive
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['naive'];
print('Observations for Naive agent loaded from library function: \n' 
       + JSON.stringify(observedStateAction) + ' \n');
var path = map(first,observedStateAction);
GridWorld.draw(world, {trajectory:path});
~~~~

~~~~
// draw_sophisticated
var world = makeRestaurantChoiceMDP();
var observedStateAction = restaurantNameToObservationTime11['sophisticated'];
print('Observations for Naive agent loaded from library function: \n' 
       + JSON.stringify(observedStateAction) + ' \n');
var path = map(first,observedStateAction);
GridWorld.draw(world, {trajectory:path});
~~~~

>**Exercise:** (Try this exercise *before* reading further). Your goal is to do preference inference from the observed actions in the codeboxes above (using only a pen and paper). The discount function is the hyperbola $$D=1/(1+kt)$$, where $$t$$ is the time from the present, $$D$$ is the discount factor (to be multiplied by the utility) and $$k$$ is a positive constant. Find a single setting for the utilities and discount function that produce the behavior in both the codeboxes above. This includes utilities for the restaurants (both *immediate* and *delayed*) and for the `timeCost` (the negative utility for each additional step walked), as well as the discount constant $$k$$. Assume there is no softmax noise. 

------

The Naive agent goes to Donut North, even though Donut South (which has identical utility) is closer to the agent's starting point. One possible explanation is that the Naive agent has a higher utility for Veg but gets "tempted" by Donut North on their way to Veg[^naive_path].

[^naive_path]: At the start, no restaurants can be reached quickly and so the agent's discount function is nearly flat when evaluating each one of them. This makes Veg look most attractive (given its higher overall utility). But going to Veg means getting closer to Donut North, which becomes more attractive than Veg once the agent is close to it (because of the discount function). Taking an inefficient path -- one that is dominated by another path -- is typical of time-inconsistent agents. 

The Sophisticated agent can accurately model what it *would* do if it ended up in location [3,5] (adjacent to Donut North). So it avoids temptation by taking the long, inefficient route to Veg. 

In this simple example, the Naive and Sophisticated agents each take paths that optimal time-consistent MDP agents (without softmax noise) would never take. So this is an example where a bias leads to a *systematic* deviation from optimality and behavior that is not predicted by an optimal model. In Chapter V.III we explore inference of preferences for time inconsistent agents.

-------
## ADD CHAPTER BREAK HERE
--------

# Formal Model and Implementation of Hyperbolic Discounting


### Formal Model of Naive and Sophisticated Hyperbolic Discounters

To formalize Naive and Sophisticated hyperbolic discounting, we make a small modificiation to the MDP agent model. The key idea is to add an additional variable for measuring time, the *delay*, which is distinct from the objective time-index (called `timeLeft` in our implementation). The objective time-index is important in planning for finite-horizon MDPs because it determines how far the agent can travel or explore before time is up. By contrast, the delays are *subjective*. They are used by the agent in *evaluating* possible future rewards but they are not an independent feature of the decision problem. (The objective time is part of the problem definition, while delays are an internal representation employed by a subset of agents). 

We use delays because discounting agents have preferences over when they receive rewards. When evaluating future prospects, they need to keep track of how far ahead in time that reward occurs, i.e. keep track of the time-delay in getting the reward. Naive and Sophisticated agents *evaluate* future rewards in the same way. They differ in how they simulate their future actions.

The Naive agent at objective time $$t$$ assumes his future self at objective time $$t+c$$ (where $$c>0$$) shares his time preference. So he simulates the $$(t+c)$$-agent as evaluating a reward at time $$t+c$$ with delay $$d=c$$ (hence discount factor $$1/1+kc$$) rather than the true delay $$d=0$$. The Sophisticated agent correctly models his $$(t+c)$$-agent future self as evaluating an immediate reward with delay $$d=0$$ and hence a discount factor of one (i.e. no discounting). 

Adding delays to our model is straightforward. In defining the MDP agent, we presented Bellman-style recursions for the expected utility of state-action pairs [TODO reference]. Discounting agents evaluate states and actions differently depending on their *delay* from the present. So we now define expected utilities of state-action-delay triples:

$$
EU[s,a,d] = \delta(d)U(s, a) + \mathbb{E}_{s', a'}(EU[s', a',d+1])
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
TODO: implement exponential discounting and show that naive/soph doesn't make a difference and agent never does the Naive path. 

As with the MDP and POMDP agents, our WebPPL implementation directly translates the mathematical formulation of Naive and Sophisticated hyperbolic discounting. The variable names correspond as follows:

- The function $$\delta$$ is named `discountFunction`

- The "perceived delay", which is the delay from which the agent's simulate future self evaluates rewards, is $$d$$ in the math and `perceivedDelay` below. 

- $$s'$$, $$a'$$, $$d+1$$ correspond to `nextState`, `nextAction` and `delay+1` respectively. 


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


// TODO - move this to a library?
var makeRestaurantUtilityFunction = function (world, rewards) { 
  return function(state, action) {
    var getFeature = world.feature;
    var feature = getFeature(state);

    if (feature.name) { return rewards[feature.name][state.timeAtRestaurant]; }
    return rewards.timeCost;
  };
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

var naiveAgent = makeAgent( 
  update(baseAgentParams, {sophisticatedOrNaive: 'naive'}), 
  world
);

// TODO: draw these trajectories. 
print('Soph traj' +  simulate(startState, world, sophisticatedAgent));
print('Naive trajectory' + 
            simulate(startState, world, naiveAgent));
~~~~
            


### Example: Procrastinating on a task

In the examples above, time-inconsistency leads to behavior that optimal agents never exhibit. However, given enough softmax noise (or some other random noise model), the Naive path will occur with non-trivial probability. If the agent goes "up" instead of "left" at $$[3,1]$$, then they will continue on to Donut North if they prefer Donuts. As we discuss in Chapter V.3, the explanation of this behavior in terms of noise becomes less likely if we see this behavior repeatedly. However, it might be unlikely that a human repeatedly (e.g. on multiple different days) takes the Naive path. So we turn to an example from everyday life where time inconsistency leads to behavior that  becomes arbitrarily unlikely on the softmax model (see refp:kleinberg2015time):

> **The Procrastination Problem**
> <br>You have a hard deadline of ten days to complete a task (e.g. write a paper for a class, complete an application or tax return). Completing the task takes a full day and has a *cost* (e.g. it's unpleasant work). After the task is complete you get a *reward* (typically exceeding the cost). There is an incentive to finish early: every day you delay finishing, your reward gets slightly smaller. (Imagine that it's good for your reputation to complete tasks early or that early applicants are considered first).

Note that if the task is worth doing at the last minute, then you should do it immediately (because the reward diminishes over time). Yet people often do this kind of task at the last minute -- the worst possible time to do it!

Hyperbolic discounting provides an elegant model of this behavior. On Day 1, a hyperbolic discounter will prefer that they complete the task tomorrow rather than today. Moreover, a Naive agent wrongly predicts they will complete the task tomorrow and so puts off the task till Day 2. When Day 2 arrives, the Naive agent reasons in the same way -- telling themself that they can avoid the work today by putting it off till tomorrow. This continues until the last possible day, when the Naive agent finally completes the task.

In this problem, the behavior of optimal and time-inconsistent agents with identical preferences (i.e. utility functions) diverges. If the deadline is $$T$$ days from the start, the optimal agent will do the task immediately and the Naive agent will do the task on Day $$T$$. [TODO: state Kleinberg result informally.]

We formalize the Procrastination Problem in terms of a deterministic graph. Suppose the **deadline** is $$T$$ steps from the start. Assume that after $$t$$ < $$T$$ steps the agent has not yet completed the task. Then the agent can take the action `"work"` (which has **work cost** $$-w$$) or the action `"wait"` with zero cost. After the `"work"` action the agent transitions to the `"reward"` state and receives $$+(R - t \epsilon)$$, where $$R$$ is the **reward** for the task and $$\epsilon$$ is how much the reward diminishes for every day of waiting (the **wait cost**). 

TODO: graph like this:

![diagram](/assets/img/diagram_procrastinate.jpg)

We simulate the behavior of hyperbolic discounters on the Procrastination Problem. We vary the discount rate $$k$$ while holding the other parameters fixed. The agent's behavior can be summarized by its final state (`"wait_state"` or `"reward_state`) and by how much time elapses before termination. When $$k$$ is sufficiently high, the agent will not even complete the task on the last day. 

~~~~
// procrastinate

// Construct Procrastinate world 
var deadline = 10;
var world = makeProcrastinationMDP(deadline);

// Agent params
var utilityTable = {reward: 5,
    waitCost: -0.1,
    workCost: -1};

var params = {utility: makeProcrastinationUtility(utilityTable),
	      alpha: 1000,
	      discount: null,
	      sophisticatedOrNaive: 'sophisticated'};

var getLastState = function(discount){
  var agent = makeHyperbolicDiscounter(update(params, {discount: discount}), 
                                       world);
  var stateActions = simulateHyperbolic(world.startState, world, agent);
  var states = map(first,stateActions);
  return [last(states).loc, stateActions.length];
};

// TODO use vegaPrint to display as table
map( function(discount){
    var lastState = getLastState(discount);
    print('Discount: ' + discount + '. Last state: ' + lastState[0] +
    '. Time: ' + lastState[1] + '\n')
}, range(8) );
~~~~


>**Exercise:**
> 1. Explain how an exponential discounter would behave on this task. Assume their utilities are the same as above and consider different discount rates.  

> 2. Run the codebox above with a Sophisticated agent. Explain the results. 



-----------

### Footnotes

