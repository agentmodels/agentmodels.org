---
layout: chapter
title: "Sequential decision problems with partial observability (POMDPs)"
description: Mathematical framework, implementation in WebPPL, Gridworld and restaurants example, bandit problems.

---

PLAN

1. Restaurant example. Human might not know which restaurants open or full. Might not know about certain restaurants. Might not know whether the restaurants have good food or have good food often. Intuition: move around world, get observatinos from world. Update beliefs based on observations, enabling better choices. Working backwards, when simulating sequences of actions, take into account possible observations and better decisions that result -- i.e. VOI of certain actions. So need to model one's future belief states. 

2. Formalizing this. Basic structure same as in MDP gridworld. Only change is an observation function that gives a dist on observations as a function of the state (i.e. agent's position). From high level, formalize as agent being uncertain about which state it is in. For instance, it might be uncertain about its location in the world and uncertain about whether a restaurant is current open or closed. Agent then receives obserations. E.g. it gets to observe its location (even if location depends on stochastic transitions). It observes whether a restaurant is open if it gets close enough. Formalization in AAAI paper. Update beliefs by simulating world given different starting states. 

3. More specialized formulation. Assume that agent faces an MDP parameterized by some variables whose values are unknown. These are variables are fixed for any given decision problem. For example, whether a restaurant is present in a particular location is fixed. The mean quality of a restaurant is fixed, etc. We call the vector of these variables the latent state. Given the latent state, the MDP has its normal state, which in gridworld is the agent's location, and we have the current observation. The agent's current location is assumed to be known without any observations. We call this kind of state the manifest state. Agent program, agent is called on a manifest state, prior distribution and obseration. Computes exp U of actions by simulating future state sequences given different possible latent states, with dist on latent states the posterior given the current observation. To simulate the world containing this agent, the world takes a state ==df {manifestState: [0,1], latentState:{donutSouthOpen: false, noodleShopOpen: true}}. The transition function and utility function depend on both the manifest and latentState. For example, if a restaurant exists, it will yield some utility and be terminal. Observations also depend on both. 

## 







~~~~
var drawGridWorld = function(world) {
  var element = wpEditor.makeResultContainer();
  GridWorld.draw(element, world);
}

var world = {
  xLim: 5,
  yLim: 5,
  blockedStates: [[0, 0], [0, 1]],
  terminals: [[2, 2], [2, 3]]
};

drawGridWorld(world);

var erp = Enumerate(function(){
    if (flip(.5)) {
       return "a";
    } else {
      if (flip(.5)) {
         return "b";
      } else {
         return "c";
      }
    }
});

viz.print(erp);
~~~~
