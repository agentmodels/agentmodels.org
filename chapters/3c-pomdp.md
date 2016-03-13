---
layout: chapter
title: "Partial observability"
description: Mathematical framework, implementation in WebPPL, Gridworld and restaurants example, bandit problems.
---


 
## Introduction: Agents with uncertainty and belief updating

The previous chapters included MDPs where the transition function is *stochastic*. This means the agent is *uncertain* about the result of taking an action in a given state. For example in Gridworld Hiking, Alice is uncertain whether she would fall down the hill if she takes the shortcut. In an MDP the agent's uncertainty cannot be altered by observation. Transitions occur according to a particular probability distribution that is fixed (with no learnable parameters). An MDP is like a fair lottery: observing the winning ticket one week does not change the distribution on tickets the following week.

In contrast, we often face problems where our uncertainty can be *reduced* by observation. In the example of Bob choosing between restaurants, Bob would not have complete knowledge of the restaurants in his neighborhood. He'd be uncertain about opening hours, the chance of getting a table, the quality of restaurants, the exact distances between locations, and so on. This uncertainty can be reduced observation: Bob can walk to a restaurant and see whether or not it's open. In other examples, the environment is stochastic but the agent can gain knowledge of the *distribution* on outcomes. For example, in Multi-arm Bandit problems, the agent learns about the distribution over rewards given by each of the arms.

To represent decision problems where the agent's uncertainty is altered by observations, we use Partially Observable Markov Decision Processes (POMDPs). We first introduce the formalism for POMDPs and then show how to extend our agent model for MDPs to an agent model that solves POMDPs. 


## Extending our agent model for POMDPs

### Informal description of POMDP agent model
- the environment now includes an observation function from states to observations.
- agent has prior uncertainty about some elements of the environment. these elements could influence observations, transitions or utilities. we focus on the case where they influence observations and transitions.
- in our examples, apart from the observation function, the environment has the same essential structure as before (including the markov assumption). previously an agent was given a state as input and had to take an action (which caused a transition). now the agent is uncertain about which state they are in. they have a prob dist b over the current state. at every time step, they update this distribution, conditioning on the observation and also on the action the agent performed last. 

- to give a concrete example, consider the example of Restaurant Choice where Bob doesn't know whether the Noodle shop is open or not. previously, the state simply consisted of Bob's location in the grid. now we think of the state as also storing whether the Noodle Shop is open (which determines whether Bob would transition to inside the Noodle Shop is he moved there from an adjacent location). If Bob knows his location (e.g. location [2,1]) but doesn't know if the Noodle Shop is open, then he has a distribution over the states [{myLocation:[2,1], Noodle Shop:'closed'}, {myLocation:[2,1], Noodle Shop:'open'}]. When Bob is close to the Noodle Shop, he will get an observation that varies depending on whether the Noodle Shop is open or closed, and so he'll rule out one of the these possible states. 

### Formal model

We first define a class of decision probems (POMDPs) and then define an agent model for optimally solving these problems (following ADD kaelbling refp:kaelbling). A Partially Observable Markov Decision Process (POMDP) is a tuple $$(S,A(s),T(s,a),U(s,a),\Omega,O$$, where:

- The components $$S$$ (state space), $$A$$ (action space), $$T$$ (transition function), $$U$$ (utility or reward function) form an MDP as defined in [chapter III.1](/chapters/3a-mdp.html), with $$U$$ assumed to be deterministic. 

- The component $$\Omega$$ is the finite space of observations the agent can receive.

- The component $$O$$ is a function  $$ O\colon S \times A \to \Delta \Omega $$. This is the *observation function*, which maps an action $$a$$ and the state $$s'$$ resulting from taking $$a$$ to an observation $$o \in \Omega$$ drawn from $$O(s',a')$$.

So at each timestep, the agent transitions from state $$s$$ to state $$s'$$ (generally unknown to the agent) having performed action $$a$$ (and where $$s' \sim T(s,a)$$). On entering $$s'$$ the agent receives an observation $$o \sim O(s',a)$$ and a utility $$U(s,a)$$. [Might be good to include influence diagram from Braziunas page 3.]. 

To characterize the behavior of an expected-utility maximizing agent, we need to formalize the belief-updating process. Let the agent's belief about their current state $$s$$ be a probability function $$b$$ over $$S$$. Then the agent's belief function $$b'$$ over their successor state is the result of a Bayesian update on the observation $$o \sim O(s',a)$$ where $$a$$ is the agent's action in $$s$$.  That is:

$$
b'(s') \propto O(s',a,o)\sum_{s \in S}{T(s,a,s')b(s)}
$$

Intuitively, the probability that $$s'$$ is the new state depends on the marginal probability of transitioning to $$s'$$ (given $$b$$) and the probability of the observation $$o$$ coming in $$s'$$. 

In our previous agent model for MDPs, we defined the expected utility of an action $$a$$ in a state $$s$$ recursively in terms of the expected utility of the resulting pair of state $$s'$$ and action $$a'$$. This same recursive characterization of expected utility still holds. The important difference is that the agent's action $$a'$$ in $$s'$$ depends on their updated belief $$b'(s')$$ given the observation they receive in $$s'$$. So the expected utility of $$a$$ in $$s$$ depends on the agent's belief $$b$$ over the state $$s$$. We call the following the *Expected Utility of State Recursion*, which defines the function $$EUS$$. This is analogous to the characterization of the *value*, $$V$$, of a state (see p109 in Kaelbling et al).

$$
EUS_{b}[s,a] = U(s,a) + E_{s',o,a'}(EUS_{b'}[s',a'_{b'}])
$$

where:
- $$s' \sim T(s,a)$$
- $$o \sim O(s',a)$$
- $$b'$$ is the updated belief function $$b$$ on $$o$$ (as defined above)
- $$a'_{b'}$$ is the softmax action the agent takes given belief $$b'$$

The agent cannot use the Expected Utility of State Recursion to directly compute the best action, since the agent doesn't know the state. Instead the agent takes an expectation over their belief distribution, picking the action $$a$$ that maximizes:

$$
EU_[b,a] = E_{s \sim b}(EUS_{b}[s,a])
$$

We can also represent the expected utility of action $$a$$ given belief $$b$$ in terms of a recursion on the successor belief state. We call this the *Expected Utility of Belief Recursion*. It's analogous to the Bellman update rule.

$$
EU_[b,a] = E_{s \sim b}( U(s,a) + E_{s',o,a'}(EU_[b',a']) )
$$

where $$s'$$, $$o$$, $$a'$$ and $$b'$$ are distributed as in the Expected Utility of State Recursion.


### Implementation of the Model
As with the agent model for MDPs, we provide a direct translation of the equations above into an agent model for solving POMDPs. The variables `nextState`, `nextObservation`, `nextBelief`, and `nextAction` correspond to $$s'$$,  $$o$$, $$b'$$ and $$a'$$ respectively, and we use Expected Utility of Belief Recursion. The following codebox displays the core `act` and `expectedUtility` functions, without defining `updateBelief`, `transition`, `observe` or `utility`. 

~~~~
var act = function(belief) {
  return Enumerate(function(){
    var action = uniformDraw(actions);
    var eu = expectedUtility(belief, action);
    factor(alpha * eu);
    return action;
  });
});

var expectedUtility = function(belief, action) {
  return expectation(
    Enumerate(function(){
      var state = sample(belief);
	  var u = utility(state, action);
	  if (state.terminateAfterAction) {
	    return u;
	  } else {
	    var nextState = transition(state, action);
	    var nextObservation = observe(nextState);
	    var nextBelief = updateBelief(belief, nextObservation, action);            
	    var nextAction = sample(act(nextBelief));   
	    return u + expectedUtility(nextBelief, nextAction);
	    }
    }));
});

// *startState* is agent's actual startState (unknown to agent)
// *priorBelief* is agent's initial belief function
var simulate = function(startState, priorBelief) {
    
  var sampleSequence = function(state, priorBelief, action) {
    var observation = observe(state);
    var belief = updateBelief(priorBelief, observation, action);
    var action = sample(act(belief));
    var output = [ [state,action] ];
      
    if (state.terminateAfterAction){
      return output;
    } else {   
      var nextState = transition(state, action);
      return output.concat(sampleSequence(nextState, belief, action));
      }
    };
  return sampleSequence(startState, priorBelief, 'startAction');
};
~~~~

To illustrate the POMDP agent in action, we implement a simplified variant of the Multi-arm Bandit Problem. In this variant, there are just two arms. Pulling an arm produces a prize (deterministically). The agent does not know initially the mapping from arms to prizes but can learn by trying the arms. In our concrete example, the first arm is known to have the prize "chocolate" and the second arm either has "champagne" or has no prize at all ("nothing").  

In our implementation of this problem, we label the two arms `[0,1]`, and use the same labels for the actions of pulling the arms. After taking action `0`, the agent transitions to a state with whatever prize is associated with `Arm0` (and gets to observe that prize). States contain properties for counting down the time (as before), as well as a `prize` property. States also contain the "latent" mapping from arms to prizes (called `armToPrize`) that determines how an agent transitions on pulling an arm.

If the agent only has one timestep in total (i.e. one bandit trial), then they will take the arm with highest expected utility (given their prior on `armToPrize`). If there are multiple trials, the agent might *explore* the lower expected utility arm (e.g. if it's maximum possible utility is higher). Try changing the number trials to see how it affects the agent's choice on the first trial. 

~~~~

// ---------------
// Defining the Bandits decision problem

// Pull arm0 or arm1
var actions = [0,1];

// use latent "armToPrize" mapping in
// state to determine which prize agent gets
var transition = function(state, action){
  return update(state, 
                {prize: state.armToPrize[action], 
                 timeLeft: state.timeLeft - 1,
                 terminateAfterAction: state.timeLeft == 2})
};

// After pulling an arm, agent observes associated prize
var observe = function(state){return state.prize;};

var startState = { prize: 'start',
                   timeLeft:3, 
                   terminateAfterAction:false,
                   armToPrize: {0:'chocolate', 1:'champagne'}
                 };
                
// ---------------
// Defining the POMDP agent

// Agent's preferences over prizes
var utility = function(state,action){
  var prizeToUtility = {chocolate: 1, nothing: 0, champagne: 1.5, start:0};
  return prizeToUtility[state.prize];
};

// Agent's prior prior includes possibility that arm1 has no prize
// (instead of champagne)
var alternativeStartState = update(startState, {armToPrize:{0:'chocolate', 1:'nothing'}});

var priorBelief = Enumerate(function(){
  return categorical( [.5, .5], [startState, alternativeStartState]);
});


// Agent's belief update: directly translates the belief update
// equation above

var updateBelief = function(belief, observation, action){
  return Enumerate(function(){
    var state = sample(belief);
    var predictedNextState = transition(state, action);
    var predictedObservation = observe(predictedNextState);
    condition(_.isEqual(predictedObservation, observation));
    return predictedNextState;
  });
};

var act = dp.cache(
  function(belief) {
    return Enumerate(function(){
      var action = uniformDraw(actions);
      var eu = expectedUtility(belief, action);
      factor(100 * eu);
      return action;
    });
  });

var expectedUtility = dp.cache(
  function(belief, action) {
    return expectation(
      Enumerate(function(){
	var state = sample(belief);
	var u = utility(state, action);
	if (state.terminateAfterAction) {
	  return u;
	} else {
	  var nextState = transition(state, action);
	  var nextObservation = observe(nextState);
	  var nextBelief = updateBelief(belief, nextObservation, action);            
	  var nextAction = sample(act(nextBelief));   
	  return u + expectedUtility(nextBelief, nextAction);
	}
      }));
  });


var simulate = function(startState, priorBelief) {
    
  var sampleSequence = function(state, priorBelief, action) {
    var observation = observe(state);
    var belief = action=='startAction' ? priorBelief : updateBelief(priorBelief, observation, action);
    var action = sample(act(belief));
    var output = [ [state,action] ];
         
    if (state.terminateAfterAction){
      return output;
    } else {   
      var nextState = transition(state, action);
      return output.concat(sampleSequence(nextState, belief, action));
    }
  };
  return sampleSequence(startState, priorBelief, 'startAction');
};

simulate(startState, priorBelief);
~~~~

PLAN:
Stochastic bandits. In two arm case, as time increases the number of possible sequences of observations blows up exponentially, so the run time should also. (What's the prior over coin weights? Shouldn't matter too much. Do we need two arms to be noisy or just one?).  

Gridworld POMDP: Same set up as before with POMDPgridworld library functions. Now we get different possible behaviors -- e.g. agent doesn't go to donut south because of ignorance or agent tries noodle and then goes on to veg. 


--------------

[Table of Contents](/)
