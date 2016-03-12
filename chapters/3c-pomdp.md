---
layout: chapter
title: "Partial observability"
description: Mathematical framework, implementation in WebPPL, Gridworld and restaurants example, bandit problems.
---

Use math formalism of the paper and from Kaelbling et al paper. Introduce simplified version of beliefAgent.wppl. Bring updateBelief into scope of agent. Simplify simulate. Might be worth discussing recursing on state vs. belief but not clear.

Should we introduce bandit example here? Yes, because we'll want to talk about it for myopic and boundVOI agents and it's good to have multiple examples. If so, showing stochastic bandits also would be ideal -- otherwise we have no stochasticity in the environment for the next few chapters. This also is a good way to introduce the intractability of POMDPs.

 
## Introduction: Agents with uncertainty and belief updating [WORK IN PROGRESS]

The previous chapters included MDPs where the transition function is *stochastic*. This means the agent is *uncertain* about the result of taking an action in a given state. For example in Gridworld Hiking, Alice is uncertain whether she would fall down the hill if she takes the shortcut. In an MDP the agent's uncertainty cannot be altered by observation. Transitions occur according to a particular probability distribution that is fixed (with no learnable parameters). An MDP is like a fair lottery: observing the winning ticket one week does not change the distribution on tickets the following week.  

In contrast, we often face problems where our uncertainty can be *reduced* by observation. In the example of Bob choosing between restaurants, Bob would not have complete knowledge of the restaurants in his neighborhood. He'd be uncertain about opening hours, chance of getting a table, restaurant quality, the exact distances between locations, and so on. This uncertainty can be reduced observation: Bob can walk to the restaurant and see whether or not it's open. In other examples, the environment is stochastic but the agent can gain knowledge of the *distribution* on outcomes. For example, in Multi-arm Bandit problems, the agent learns about the distribution over rewards given by each of the arms.

To represent decision problems where the agent's uncertainty is altered by observations, we use Partially Observable Markov Decision Processes (POMDPs). We first introduce the formalism for POMDPs and show how to extend our agent model that solves MDPs to an agent that solves POMDPs. 


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





~~~~

// generalization of *mdpSimulate* from previous chapters

var pomdpSimulate = function(startState, actualTotalTime, 
                              perceivedTotalTime, params){

  // Key functions defining POMDP
  var utility = params.utility;
  var transition = params.transition;
  var observe = params.observe;
  var observationEquality = params.observationEquality;

  // Constructor for states
  var buildState = function(manifestState,latentState){
    return {manifestState:manifestState, latentState:latentState};
  };

  // Takes agent's belief ERP and updates it on a single observation.
  // Since *observe* takes a state (not a latentState) 
  // we need to build a state from the sampled latentState. 
  var updateBelief = dp.cache(
    function(currentBelief, manifestState, observation, params){

      return Enumerate( function(){
        var hypotheticalLatentState = sample(currentBelief);
        var hypotheticalState = buildState(manifestState, hypotheticalLatentState);
        var hypotheticalObservation = observe(hypotheticalState, params);
        condition( observationEquality(hypotheticalObservation, observation) );
        return hypotheticalLatentState;
      });
    });
  

  // Agent is called on *manifestState* not *state* 
  // since he doesn't know his *state*
  
  var agent = dp.cache( 
    function(manifestState, timeLeft, currentBelief, observation, params){
    
    return Enumerate( function(){
      var updatedBelief = updateBelief(currentBelief, manifestState,
                                       observation);
      var action = uniformDraw(params.actions);
      
      var expectedUtility = expectation(
        Enumerate(function(){
          var state = buildState(manifestState, sample(updatedBelief));
          return expUtility(state, action, timeLeft, updatedBelief, params);   
        }));
      
      factor(params.alpha * expectedUtility);
      return {action: action, belief: updatedBelief};
    });
  });
  
  
  var expUtility = dp.cache(
    function(state, action, timeLeft, currentBelief, params){ 
      var u = utility(state, action, params);
      
      if (timeLeft - 1 == 0){
        return u;
      } else {                     
        return u + expectation( Enumerate(function(){
          var nextState = transition(state, action, params);
          var nextManifestState = getManifestState(nextState,params);
          var observation = observe(nextState,params);
          var out = sample(agent(nextManifestState, timeLeft-1, 
                                 currentBelief, observation, params));
          var nextAction = out.action;
          var nextBelief = out.belief;
          return expUtility(nextState, nextAction, timeLeft-1, nextBelief, params);
        }));
      }                      
    });
  

  var simulate = function(startState, actualTotalTime, perceivedTotalTime, params){
    
    var sampleSequence = function(state, actualTimeLeft, perceivedTimeLeft, 
                                   history, currentBelief, observation){
      
      if (actualTimeLeft==0){
        return history.slice(0,history.length-1);
      } else {
        
        var out = sample(agent(getManifestState(state, params), perceivedTimeLeft,
                               currentBelief, observation, params));
        var action = out.action;
        var updatedBelief = out.belief;
        var nextState = transition(state, action, params);
        var nextObservation = observe(nextState, params);
        var nextHistory = push(history, nextState);
        
        return sampleSequence(nextState, actualTimeLeft-1, perceivedTimeLeft-1, 
                              nextHistory, updatedBelief, nextObservation);
      }
    };
    
    return Enumerate(function(){    
      var startHistory = [startState];
      var observation = observe(startState, params);
      var latentStatePrior = params.latentStatePrior;
      return sampleSequence(startState, actualTotalTime, perceivedTotalTime, 
                            startHistory, latentStatePrior, observation);
      
    });                 
  };
  
  return simulate(startState, actualTotalTime, perceivedTotalTime, params);
};
pomdpSimulate;
~~~~



--------------

[Table of Contents](/)
