---
layout: chapter
title: "Reinforcement Learning to Learn MDPs"
description: RL vs. optimal Bayesian approach to Bandits, Softmax Greedy, Posterior Sampling for Bandits and MDPs, Q-learning and Policy Gradient
---

## Introduction

The previous chapter introduced POMDPs: decision problems where some features of the environment are initially unknown to the agent but can be learned by observation. We showed how to compute optimal Bayesian behavior for POMDPs. Unfortunately, this computation is infeasible for all but the simplest POMDPs. In practice, many POMDP problems can be solved heuristically using "Reinforcement Learning" (RL). RL algorithms are conceptually simple, scalable and effective both in discrete and continuous state spaces. They are central to achieving state-of-the-art performance in sequential decision problems in AI, including playing Go [cite], playing videogames from raw pixels [cite], and continuous control for robotics [cite]. 

## Reinforcement Learning for Bandits
The previous chapter showed how the optimal POMDP agent solves Bandit problems. Here we apply Reinforcement Learning to Bandits.

In the POMDP chapter, we modeled the agent's uncertainty as being over the transition transition function of their environment; their utility function on states was always known. In this chapter, we model the agent as initially uncertain about the MDP they are in. They can be uncertain about both the utility and transition functions. The definition of an MDP is the same as <a href="/chapters/3a-mdp.html#mdp">before</a> but we sometimes follow the RL convention and say "reward function" instead of "utility function". 


### Softmax Greedy Agent
We start with a "greedy" agent with softmax noise, which is similar to the well-known "Epsilon Greedy" agent for Bandits. The Softmax Greedy agent updates beliefs about the hidden state (the expected rewards for the arms) using Bayesian updates (as with the optimal POMDP agent). Yet instead of making sequential plans that balance exploration (e.g. making informative observations) with exploitation (gaining high reward), the Greedy agent takes the action with highest *immediate* expected return[^greedy].

Here we implement the Greedy agent on Bernoulli Bandits, where each arm is a Bernoulli distribution with a fixed coin-weight. We measure the agent's performance by computing the *cumulative regret* over time. The regret for an action is the difference in expected returns between the action and the objective best action[^regret].

[^greedy]: The standard Epsilon/Softmax Greedy agent maintains point estimates for the expected rewards of the arms. In WebPPL it's natural to use distributions instead. In a later chapter, we implement a more general Greedy/Myopic agent by extending the POMDP agent. [TODO link]. 

[^regret]:The regret is a standard Frequentist metric for performance. Bayesian metrics, which take into account the agent's priors, can also be defined but are beyond the scope of this chapter. 

~~~~
///fold:
var cumsum = function (xs) {
  var acf = function (n, acc) { return acc.concat( (acc.length > 0 ? acc[acc.length-1] : 0) + n); }
  return reduce(acf, [], xs.reverse());
  }
  
///
  

// Define Bandit problem

// Pull arm0 or arm1
var actions = [0, 1];

// Given a state (a coin-weight p for each arm), sample reward
var observeStateAction = function(state, action){
  var armToCoinWeight = state;
  return sample(Bernoulli({p : armToCoinWeight[action]})) 
};


// Greedy agent for Bandits
var makeGreedyBanditAgent = function(params) {
  var priorBelief = params.priorBelief;

  // Update belief about coin-weights from observed reward
  var updateBelief = function(belief, observation, action){
    return Infer({ model() {
      var armToCoinWeight = sample(belief);
      condition( observation === observeStateAction(armToCoinWeight, action))
      return armToCoinWeight;
    }});
  };
  
  // Evaluate arms by expected coin-weight
  var expectedReward = function(belief, action){
    return expectation(Infer( { model() {
      var armToCoinWeight = sample(belief);
      return armToCoinWeight[action];
    }}))
  }

  // Choose by softmax over expected reward
  var act = dp.cache(
    function(belief) {
      return Infer({ model() {
        var action = uniformDraw(actions);
        factor(params.alpha * expectedReward(belief, action))
        return action;
      }});
    });

  return { params, act, updateBelief };
};

// Run Bandit problem
var simulate = function(armToCoinWeight, totalTime, agent) {
  var act = agent.act;
  var updateBelief = agent.updateBelief;
  var priorBelief = agent.params.priorBelief;

  var sampleSequence = function(timeLeft, priorBelief, action) {
    var observation = observeStateAction(armToCoinWeight, action);
    var belief = ((action === 'noAction') ? priorBelief :
                  updateBelief(priorBelief, observation, action))
    var action = sample(act(belief));

    return (timeLeft === 0) ? [action] : 
    [action].concat(sampleSequence(timeLeft-1, belief, action));
  };
  return sampleSequence(totalTime, priorBelief, 'noAction');
};


// Agent params
var alpha = 30
var priorBelief = Infer({  model () {
  var p0 = uniformDraw([.1, .3, .5, .6, .7, .9]);
  var p1 = uniformDraw([.1, .3, .5, .6, .7, .9]);
  return { 0:p0, 1:p1};
} });

// Bandit params
var numberTrials = 500;
var armToCoinWeight = { 0: 0.5, 1: 0.6 };

var agent = makeGreedyBanditAgent({alpha, priorBelief});
var trajectory = simulate(armToCoinWeight, numberTrials, agent);

// Compare to random agent
var randomTrajectory = repeat(
    numberTrials, 
    function(){return uniformDraw([0,1]);}
);

// Compute agent performance
var regret = function(arm) { 
  var bestCoinWeight = _.max(_.values(armToCoinWeight))
  return bestCoinWeight - armToCoinWeight[arm];
};
 
var trialToRegret = map(regret,trajectory);
var trialToRegretRandom = map(regret, randomTrajectory)
var ys = cumsum( trialToRegret) 

print('Number of trials: ' + numberTrials);
print('Total regret: [GreedyAgent, RandomAgent]  ' + 
      sum(trialToRegret) + ' ' + sum(trialToRegretRandom))
print('Arms pulled: ' +  trajectory);

viz.line(_.range(ys.length), ys, {xLabel:'Time', yLabel:'Cumulative regret'});
~~~~

How well does the Greedy agent do? It does best when the difference between arms is large but does well even when the arms are close. Greedy agents perform well empirically on a wide range of Bandit problems [cite precup] and if their noise decays over time they can achieve asymptotic optimality [cite cesa-bianchi]. In contrast to the optimal POMDP agent from the previous chapter, the Greedy Agent scales well in both the arms and trials. Given that the Greedy Agent converges on the good performance quickly, why would anyone be interested in the POMDP solution? We defer this question to the appendix [TODO link]. 


>**Exercises**:

> 1. Modify the code above so that it's easy to repeatedly run the same agent on the same Bandit problem. Compute the mean and standard deviation of the agent's total regret averaged over 20 episodes on the Bandit problem above. Use WebPPL's library [functions](http://docs.webppl.org/en/master/functions/arrays.html). 

> 2. Set the softmax noise to be low. How well does the Greedy Softmax agent do? Explain why. Keeping the noise low, modify the agent's priors to be overly "optimistic" about the expected reward of each arm (without changing the support of the prior distribution). How does this optimism change the agent's performance? Explain why. (An optimistic prior assigns a high expected reward to each arm. This idea is known as "optimism in the face of uncertainty" in the RL literature.)
> 3. Modify the agent so that the softmax noise is low and the agent has a "bad" prior (i.e. one that assigns a low probability to the truth) that is not optimistic. Will the agent always learn the optimal policy (eventually?) If so, after how many trials is the agent very likely to have learned the optimal policy? (Try to answer this question without doing experiments that take a long time to run.)


### Posterior Sampling
Posterior sampling (or "Thompson sampling") is the basis for another algorithm for Bandits. This algorithm generalizes to arbitrary discrete MDPs, as we show below. The Posterior-sampling agent updates beliefs using standard Bayesian updates. Before choosing an arm, it draws a sample from its posterior on the arm parameters and then chooses greedily given the sample. In Bandits, this is similar to Softmax Greedy but without the softmax parameter $$\alpha$$.

>**Exercise**:
> Implement Posterior Sampling for Bandits by modifying the code above. (You only need to modify the `act` function.) Compare the performance of Posterior Sampling to Softmax Greedy (using the value for $$\alpha$$ in the codebox above). You should vary the `armToCoinWeight` parameter and the number of arms. Evaluate each agent by computing the mean and standard deviation of rewards averaged over many trials. Which agent is better overall and why?

<!-- TODO maybe we should include this code so casual readers can try it? -->

<!-- Modified act function:
var act = dp.cache(
    function(belief) {
      var armToCoinWeight = sample(belief);  // sample coin-weights
      return Infer({ model() {
        var action = uniformDraw(actions);
        factor(1000 * armToCoinWeight[action])  // pick arm with max weight
        return action;
      }});
    });
-->

-----------

## RL algorithms for MDPs
The Bandit problem is a one-state MDP. We now consider RL algorithms for learning discrete MDP's with any number of states. Algorithms are either *model-based* or *model-free*:

1. *Model-based* algorithms learn an explicit representation of the MDP's transition and reward functions. These representations are used to compute a good policy. 

2. *Model-free* algorithms do not explicitly represent the transition and reward functions. Instead they explicitly represent either a value function (e.g. an estimate of the $$Q^*$$-function) or policy. 

### Q-learning (TD-learning)
Q-learning is the best known RL algorithm and is model-free. A Q-learning agent stores and updates a point estimate of the expected utility of each action under the optimal policy (i.e. an estimate $$\hat{Q}(s,a)$$ for $$Q^*(s,a)$$). Provided the agent takes random exploratory actions, these estimates converge in the limit (cite Watkins). In our framework, it's more natural to implement *Bayesian Q-learning* (Dearden et al), where the point estimates are replaced with Bayesian posteriors.

The defining property of Q-learning (as opposed to SARSA or Monte-Carlo) is how it updates its Q-value estimates. After each state transition $$(s,a,r,s')$$, a new Q-value estimate is computed: <br>

$$
\hat{Q}(s,a) = r + \max_{s'}{Q(s',a')}
$$


CODEBOX: Bayesian Q-learning. Apply to gridworld where goal is to get otherside of the and maybe there are some obstacles. For small enough gridworld, POMDP agent will be quicker.

Note that Q-learning works for continuous state spaces. 

### Policy Gradient

TODO

<!-- - Directly represent the policy. Stochastic function from states to actions. (Can put prior over that the params of stochastic function. Then do variational inference (optimization) to find params that maximize score.)

Applied to Bandits. The policy is just a multinomial probability for each arm. You run the policy. Then take gradient in direction that improves the policy. (Variational approximaton will be exact in this case.) Gridworld example of get from top left to bottom right (not knowing initially where the goal state is located). You are learning a distribution over actions in these discrete location. So you have a multinomial for each state.
-->

### Posterior Sampling Reinforcement Learning (PSRL)

Posterior Sampling Reinforcemet Learning (PSRL) is a model-based algorithm that generalizes posterior-sampling for Bandits to discrete, finite-horizon MDPs (cite Strens). The agent is initialized with a Bayesian prior distribution on the reward function $$R$$ and transition function $$T$$. At each episode the agent proceeds as follows:

> 1. Sample $$R$$ and $$T$$ (a "model") from the distribution. Compute the optimal policy for this model and follow it until the episode ends.
> 2. Update the distribution on $$R$$ and $$T$$ on observations from the episode.

Intuition for PSRL: if very confident, agent mainly exploit a model. If unconfident then will act as if different models are true. if one plausible model says that certain states have high reward when they in fact don't, agent will sample that model and visit those states and discover that they suck. after this, the agent will update and won't consider those models again.

Our implementation. The PSRL agent is simple to implement in our framework. The prior and belief-updating re-uses code from the POMDP case: $$R$$ and $$T$$ are treated as latent state and are observed every state transition. Every episode, an MDP agent chooses actions by planning in the sampled model. Since the sampled model can differ radically from the model the agent is actually in, the agent may observe very incongruous state transitions. <!-- Could also note that the agent's model must have the actual transitions in its support at each timestep. -->

TODOS: <br>
Gridworld maze example is unknown transition function. So requires a change to code below (which assumes same transitions for agent and simulate function. Clumpy reward uses same model below but has rewards be correlated. Should be easy to implement a simple version of this. Visualization should depict restaurants (which have non-zero rewards). 

Gridworld maze: Agent is in a maze in perfect darkness. Each square could be wall or not with even probability. Agent has to learn how to escape. Maze could be fairly big but want a fairly short way out. Model for T.

Clumpy reward model. Gridworld with hot and cold regions that clump. Agent starts in a random location. If you assume clumpiness, then agent will go first to unvisited states in good clumps. Otherwise, when they start in new places they'll explore fairly randomly. Could we make a realistic example like this? (Once you find some bad spots in one region. You don't explore anywhere near there for a long time. That might be interesting to look at. Could have some really cold regions near the agent.

Simple version: agent starts in the middle. Has enough time to go to a bunch of different regions. Regions are clumped in terms of reward. Could think of this a city, cells with reward are food places. There are tourist areas with lots of bad food, foodie areas with good food, and some places with not much food. Agent without clumping tries some bad regions first and keeps going back to try all the places in those regions. Agent with clumping tries them once and then avoids. [Problem is how to implement the prior. Could use enumeration but keep number of possibilities fairly small. Could use some approximate method and just do a batch update at the end of each episode. That will require some extra code for the batch update.]







~~~~
///fold:

// Construct Gridworld (transitions but not rewards)
var ___ = ' '; 

var grid = [
  [ ___, ___, ___],
  [ ___, ___, ___],  
  [ ___, ___, ___]
];

var pomdp = makeGridWorldPOMDP({
  grid,
  start: [0, 0],
  totalTime: 5
});

var transition = pomdp.transition

var actions = ['l', 'r', 'u', 'd'];

var utility = function(state, action) {
  var loc = state.manifestState.loc;
  var r = state.latentState.rewardGrid[loc[0]][loc[1]];
  
  return r;
};


// Helper function to generate agent prior
var getOneHotVector = function(n, i) {
  if (n==0) { 
    return [];
  } else {
    var e = 1*(i==0);
    return [e].concat(getOneHotVector(n-1, i-1));
  }
};
///

var observeState = function(state) { 
  return utility(state);
};




var makePSRLAgent = function(params, pomdp) {
  var utility = params.utility;

  // belief updating: identical to POMDP agent from Chapter 3c
  var updateBelief = function(belief, observation, action){
    return Infer({ model() {
      var state = sample(belief);
      var predictedNextState = transition(state, action);
      var predictedObservation = observeState(predictedNextState);
      condition(_.isEqual(predictedObservation, observation));
      return predictedNextState;
    }});
  };

  // this is the MDP agent from Chapter 3a
  var act = dp.cache(
    function(state) {
      return Infer({ model() {
        var action = uniformDraw(actions);
        var eu = expectedUtility(state, action);
        factor(1000 * eu);
        return action;
      }});
    });

  var expectedUtility = dp.cache(
    function(state, action) {
      return expectation(
        Infer({ model() {
          var u = utility(state, action);
          if (state.manifestState.terminateAfterAction) {
            return u;
          } else {
            var nextState = transition(state, action);
            var nextAction = sample(act(nextState));
            return u + expectedUtility(nextState, nextAction);
          }
        }}));
    });

  return { params, act, expectedUtility, updateBelief };
};


//NOTES:
//We simulate with a single agent. Whenever the agent takes actions,
//they are given a state (*believedState*) that contains the *latentState*.
//Since utility(s,a) just depends on *latentState.rewardGrid*, this is equivalent to giving
//them a reward function. If the agent is given the same starting state twice,
//then their old plans are re-used due to caching.

//(It might be a bit clearer to create the agent anew every episode. On the
//other hand, the current code is elegant and exploits caching.)

//People just reading this chapter will proably be confused by manifest/latent. 
//(We could bring the presentation closer to standard model-based RL
//where R and T are unknown. But it's not clear it's worth doing so.)



var simulatePSRL = function(startState, agent, numEpisodes) {
  var act = agent.act;
  var updateBelief = agent.updateBelief;
  var priorBelief = agent.params.priorBelief;

  var runSampledModelAndUpdate = function(state, priorBelief, numEpisodesLeft) {
    var sampledState = sample(priorBelief);
    var trajectory = simulateEpisode(state, sampledState, priorBelief, 'noAction');
    var newBelief = trajectory[trajectory.length-1][2];
    var newBelief2 = Infer({ model() {
      return extend(state, {latentState : sample(newBelief).latentState });
    }});
    var output = [trajectory];

    if (numEpisodesLeft <= 1){
      return output;
      } else {
      return output.concat(runSampledModelAndUpdate(state, newBelief2,
                                                    numEpisodesLeft-1));
    }
  };

  var simulateEpisode = function(state, sampledState, priorBelief, action) {
    var observation = observeState(state);
    var belief = ((action === 'noAction') ? priorBelief : 
                  updateBelief(priorBelief, observation, action));

    var believedState = extend(state, { latentState : sampledState.latentState });
    var action = sample(act(believedState));
    var output = [[state, action, belief]];

    if (state.manifestState.terminateAfterAction){
      return output;
    } else {
      var nextState = transition(state, action);
      return output.concat(simulateEpisode(nextState, sampledState, belief, action));
    }
  };
  return runSampledModelAndUpdate(startState, priorBelief, numEpisodes);
};


// Construct agent prior belief

// Combine manifest (fully observed) state with prior on latent state
var getPriorBelief = function(startManifestState, latentStateSampler){
  return Infer({ model() {
    return {
      manifestState: startManifestState, 
      latentState: latentStateSampler()};
  }});
};

// True reward function
var trueLatentReward = {
  rewardGrid : [
      [ 0, 0, 0],
      [ 0, 0, 0],
      [ 0, 0, 1]
    ]
};

// True start state
var startState = {
  manifestState: { 
    loc: [0, 0],
    terminateAfterAction: false,
    timeLeft: 5
  },
  latentState: trueLatentReward
};

// Agent prior on reward functions
var latentStateSampler = function() {
  var flat = getOneHotVector(9, randomInteger(9));
  return { 
    rewardGrid : [flat.slice(0,3), flat.slice(3,6), flat.slice(6,9)] 
  };
}

var priorBelief = getPriorBelief(startState.manifestState, latentStateSampler);

// Build agent (using 'pomdp' object defined above fold)
var agent = makePSRLAgent({ utility, priorBelief, alpha: 100 }, pomdp);

var numEpisodes = 10
var trajectories = simulatePSRL(startState, agent, numEpisodes);

var project = function(x) { return first(x).manifestState.loc; };
var s = map(function (t) { return map(project, t); }, trajectories)
print(s)

var plotManifest = function(trajectory) { 
  var manifestStates = map(function(tuple) { return tuple[0].manifestState; }, trajectory);
  viz.gridworld(pomdp.MDPWorld, { trajectory: manifestStates });
};
plotManifest(trajectories[0]);
plotManifest(trajectories[1]);
plotManifest(trajectories[2]);
plotManifest(trajectories[3]);
plotManifest(trajectories[4]);
plotManifest(trajectories[5]);
plotManifest(trajectories[6]);
plotManifest(trajectories[7]);
plotManifest(trajectories[8]);
plotManifest(trajectories[9]);
~~~~



----------

## Appendix: POMDP agent vs. RL agent

First consider the Bandit problem. The POMDP agent is slow (polynomial in number of trials and exponential in # arms? ). The RL agent is almost always used in practical Bandit problems. The optimal POMDP agent solves a harder problem. It computes what to do for any possible sequence of observations. This means the POMDP agent, after computing a policy once, could immediately take the optimal action given any sequence of observations without doing any more computation. By contrast, RL agents store information only about the present Bandit problem -- and in most Bandit problems this is all we care about. 

General MDPs. The previous chapter introduced a POMDP version of the Restaurant problem in Gridworld, where the agent doesn't know initially if each restaurant is open or closed. How would RL agents compare to POMDP agents on this problem?

One way to think of the Restaurant Choice problem is as an *Episodic* POMDP. At the start of each episode, the agent is uncertain about which restaurants are open or closed. Over repeated episodes, they learn about the *distribution* on restaurants being open but they never know for sure (since restaurants might close down or vary their hours) and so they may need to update their beliefs on any given episode. (A similar example in the POMDP literature is the Tiger Problem.) This kind of problem is ill-suited to standard RL algorithms. Such algorithms assume that the hidden state is an MDP that is fixed across all episodes. POMDP algorithms, on the other hand, take into account the fact that there is new (but observable) hidden state every episode.

<!-- The general learning problem: there is some state that's initially unknown and fixed across episodes and some state that's random across episodes but observable. A POMDP agent should be able to learn both of these -->

Alternatively, we could think of the Restaurant Choice problem as an episodic MDP. Initially, the agent doesn't know which restaurants are open. But once they find out there is nothing more to learn: the same restaurants are open each episode. In this kind of example, RL techniques work well and are typically what's used in practice. 

<!--
Table:

Structure given / unknown        MDP                                  POMDP
 KNOWN                        Planning (Solve exactly DP)       POMDP solve (Belief MDP)
 LEARNED                      POMDP solver (exact Bayesian), RL    POMDP solve
 -->



<!-- ### RL and Inferring Preferences

Most IRL is actually inverse planning in an MDP. Assumption is that it's an MDP and human already knows R and T. Paper on IRL for POMDPs: assume agent knows POMDP structure. Much harder inference problem. 

We have discussion of biases that humans have: hyperbolic discounting, bounded planning. These are relevant even if human knows structure of world and is just trying to plan. But often humans don't know structure of world. Better to think of world as RL problem where MDP or POMDP also is being learned. Problem is that there are many RL algorithms, they generally involve lots of randomness or arbitrary parameters. So hard to make precise predictions. Need to coarsen. Show example of this with Thompson sampling for Bandits. 

Could discuss interactive RL. Multi-agent case. It's beyond scope of modeling.
-->




### Footnotes

