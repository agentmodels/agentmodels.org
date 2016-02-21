## Coding Goals. Deadline Tuesday 23rd Feb

###1. POMDP: Generative model for Gridworld and multi-arm bandits.
It seems valuable to do the standard stochastic bandits. This requires a modification of our code (which assumes deterministic rewards and doesn't do belief updating on the basis of rewards). Stochastic bandits will be slow but we can at least show some small examples. 

###2. Inference on trajectories for POMDP agents.
Bandits and gridworld examples. In Gridworld, you can explain Naive and Sophisticated trajectories in terms of false beliefs about Donut and Noodle respectively. For bandits, you infer the agent's prior belief about the utility of an arm by whether they try the arm. Daniel has been working towards this. One thing is to use agent/world factoring from John's hyperbolic code. 

###3. Hyperbolic discounting agent generative model and inference
Have it run fast on gridworld with the naive and sophisticated behavior. John is close to finishing here. (Need to make code part of src, add comments and clearer variable names, and also have documented library functions for constructing gridworld and hyperbolic agent). 

###4. Myopic and bounded-VOI generative model
Extending the model for the hyperbolic-discounting agent. Show how myopic agents do less exploration in bandit problems.

###5. Integrate pomdp and hyperbolic generative model
Are there distinctive behaviors of this model on the restaurant problem? In bandit problems, hyperbolic agent will explore less due to discounting. 

###6. Inference in POMDP + hyperbolic model on Gridworld
This should result in multi-modal explanations of Naive and Sophisticated scenarios for the donut problem. (Show strength of inference of the different explanations. The belief and preference based explanations are more plausible because they need less specific parameter settings).

###7. Reproduce graphs from NIPS paper.
These compare inferences of the flexible model (which includes biases) vs. the optimal model. (We should be able to improve on this by having a broader flexible model than in the NIPS paper).

###8. Visualizations
For the various examples, we need the Gridworld showing the trajectory we condition on and the graph (using viz.print) showing comparing the optimal model to our flexible model. 



## Specific plans: Sunday 21 Feb

### Hyperbolic (John)
Set Digest == 2 and do big donut world.

### Generative model
In addition to examples showing Naive and Sophisticated behavior, show the changing expected utilities for the Naive agent, which leads to the time inconsistency. (Ideally we would show these dynamically update -- maybe just showing utilities for the square next to donutNorth).

### Inference from naive and sophisticated scenarios.

Example 1 shows the posterior on donut and veg for naive agent (can fix all parameters apart from a single utility parameter for donut and veg to make inference fast) for a fine grid (as shown in the AAAI paper).

Example 2 includes inference on all the parameters. This should show that from a single trajectory, the model can infer either the naive or sophisticated explanations (rather than just assuming high noise). This will need MCMC. Can simplify inference a bit by assuming the noodle is bad (as the main conceptual point here is explanations involving high alpha vs. explanations involving nonzero discount rate and naive/sophisticated planning). The prior on the *discount* should assign a high probability to zero discounting (while the other priors should be broad/uniform as possible). 

Example 3 shows inference from multiple trajectories. This should make the high alpha (random noise) explanation less likely -- assuming we see behavior consistent with hyperbolic discounting both times. One case is where you see the same trajectory (e.g. Naive) twice. Another is you first see the Naive trajectory and then you see the agent starting to the right of noodle and going up and then left to veg. A final example is one where the route directly north to veg/donutNorth is blocked (so the world is actually different). Here the Naive agent would take the long route. So if you see this trajectory and the normal Naive trajectory, this is good evidence for Naive over a noise-based explanation. 


(If we have time on Tuesday, we could implement the procrastination problem). 

--------

### POMDP (Daniel).

## Generative model for pomdp

Specialize the beliefDelayAgent to a "beliefAgent" by getting rid of the delays but keeping everything else the same. Write bandit tests for this agent (basically the tests currently in `beliefDelayAgent.wppl`).

On the same model as the bandit tests, write code for the pomdp version of gridworld (which has an `observe` function and `manifestStateToAction`. This should re-use the `newGridworld` code as much as possible. Daniel should chat with John about the design here and then write a first version. (We can chat about that version when it's done). For now, the `latentState` will just be {donutSouthOpen: true, noodleOpen: false}. The transition function checks whether a restaurant is closed and blocks the agent from moving to the restaurant if it is closed. (Apart from this, it's the same transition function as in the MDP). The agent observes that the restaurant is closed when in the adjacent locations. The function `manifestStateToActions` can allow the agent the action that would move it to the closed restaurant, even though this action won't do anything. (This is simpler than having the available actions depend on the agent's beliefs). 

Once the gridworld code is ready, show examples with the 'Naive' trajectory (where the agent has a high prior that `donutSouthOpen:false`) and with the 'Sophisticated' trajectory (where the agent believes Noodle is open but it's actually closed). 

## Inference for POMDP
Show a simple bandit examples. First, with k arms. Agent is uncertain about value of arms. We don't know his expected values for these arms. If he explores then he thinks EV is high. Otherwise, he'll just stay with the arm he knows.

Then show inference in gridworld for the naive and sophisticated trajectories. The hardest case (where we infer preferences, alpha and beliefs) will probably need MCMC. 


 









